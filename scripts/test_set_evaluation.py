"""
Paper evaluation of an external test set: one tissue type, one slide per patient,
all slides non-healthy (e.g. lymphoma). Scored by one trained ABMIL checkpoint,
using the saved ABMILInference cache where possible.

    import test_set_evaluation as ev

    # 1. Detection: share of slides called non-healthy, per diagnosis
    ext = ev.load_predictions(inference_cache, slides, meta, "path", "M_idx", "rekvnr", "diagnose")
    ev.table_detection(ext, "diagnose")

    # 2. Faithfulness of the two ROI arms: deletion curves on P(non-healthy)
    dele = ev.run_deletion_curves(slides, abmil_path, zarr_dir)
    ev.table_deletion(dele)
    ev.fig_deletion_curves(dele)

    # 3. Model evidence per diagnosis: P(non-healthy), disease-like tiles, concentration
    evid = ev.evidence_per_slide(inference_cache, ext, abmil_path)
    ev.table_evidence_by_diagnosis(evid, "diagnose")
    ev.fig_evidence_by_diagnosis(evid, "diagnose")
"""

import os
import pickle
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats
from tqdm import tqdm

import cv_paper_results as cv
from paper_style import MM, PALETTE, INK, MUTED
from abmil import load_checkpoint, _load_slide_feats, get_tile_contributions

EVIDENCE = {
    "prob_pos": "P(non-healthy)",
    "frac_disease_like": "Fraction of disease-like tiles",
    "frac_tiles_half_contribution": "Fraction of tiles giving\n50% of contribution",
}
# Deletion rankings, matching ROISelector's defaults: the disease arm takes the highest
# contribution_score, the healthy arm the lowest contrast_score.
DELETION = {
    "disease": ("Most disease-driving first (disease arm)", PALETTE[1]),
    "healthy": ("Most healthy-like first (healthy arm)", PALETTE[2]),
    "random": ("Random", MUTED),
}
DEFAULT_FRAC_GRID = np.array([0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])


@lru_cache(maxsize=1)
def _slide_cache(cache_path):
    """ABMILInference cache {slide_path: slide_data}, loaded once."""
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    return cached.get("slide_cache", cached)


def _name(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def _groups(df, group_col, min_n):
    """Group labels with fewer than min_n slides merged into 'Other', ordered by size."""
    counts = df[group_col].value_counts()
    g = df[group_col].where(df[group_col].map(counts) >= min_n, "Other")
    order = [x for x in g.value_counts().index if x != "Other"] + (["Other"] if (g == "Other").any() else [])
    return g, order


def _mean_ci(x):
    """Mean and t-based 95% CI of the mean."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


# ----------------------------------------
# 1. Detection
# ----------------------------------------

def load_predictions(cache_path, slides, meta, filename_col, label_col, patient_col, group_col):
    """One row per slide (slide, group, patient, label, prob_pos, pred, n_tiles). Reuses cv.from_cache."""
    return cv.from_cache(cache_path, slides, meta, filename_col, label_col, patient_col, group_col)


def table_detection(ext, group_col, min_n=5):
    """
    Slides called non-healthy (P(non-healthy) >= 0.5) per diagnosis and overall.

    All slides are truly non-healthy, so this proportion is the sensitivity; AUC and
    specificity are undefined without healthy slides. The 95% CI is the exact
    (Clopper-Pearson) binomial interval for that proportion: the range of true
    detection rates compatible with k of n slides called non-healthy. It describes
    uncertainty in the detection rate, not in any single slide's P(non-healthy).
    Slides count as independent trials because there is one slide per patient.
    Diagnoses with fewer than min_n slides are merged into 'Other'.
    """
    g, order = _groups(ext, group_col, min_n)
    rows = []
    for grp in order + ["All"]:
        d = ext if grp == "All" else ext[g == grp]
        k, n = int(d["pred"].sum()), len(d)
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        rows.append({group_col: grp, "Slides": n, "Called non-healthy": k,
                     "Sensitivity, % (95% CI)": f"{100 * k / n:.1f} ({100 * ci.low:.1f}–{100 * ci.high:.1f})"})
    print("Sensitivity = share of slides with P(non-healthy) >= 0.5; 95% CI: exact binomial (Clopper-Pearson), "
          "one slide per patient.")
    return pd.DataFrame(rows)


# ----------------------------------------
# 2. Deletion curves
# ----------------------------------------

@torch.no_grad()
def _slide_deletion(model, feats, frac_grid, n_random, rng):
    """P(non-healthy) after deleting the top fraction of tiles per ranking (see DELETION)."""
    n = feats.shape[0]
    c = get_tile_contributions(model, feats)
    orders = {"disease": np.argsort(-c["contribution_score"].numpy()),
              "healthy": np.argsort(c["contrast_score"].numpy())}

    def p_keep(keep):
        keep = torch.as_tensor(np.sort(keep), device=feats.device)
        return float(torch.softmax(model(feats[keep])["logits"].float(), dim=0)[1])

    rows = []
    for frac in frac_grid:
        n_del = min(int(round(frac * n)), n - 1)  # always keep at least one tile
        probs = {k: p_keep(o[n_del:]) for k, o in orders.items()}
        probs["random"] = float(np.mean([p_keep(rng.permutation(n)[n_del:]) for _ in range(n_random)]))
        rows += [{"frac_deleted": frac, "ranking": k, "prob_pos": p} for k, p in probs.items()]
    return rows


def run_deletion_curves(slides, checkpoint_path, zarr_dir, frac_grid=DEFAULT_FRAC_GRID, n_random=5, seed=0):
    """
    Delete tiles in three orders and track P(non-healthy) after each step:
      disease: highest contribution_score first (what the disease arm cuts) -> P should drop.
      healthy: lowest contrast_score first (what the healthy arm cuts)      -> P should rise or stay.
      random:  baseline, mean of n_random permutations.
    Returns a long table (slide, frac_deleted, ranking, prob_pos).
    """
    model, config, _ = load_checkpoint(checkpoint_path)
    device = next(model.parameters()).device
    rng = np.random.RandomState(seed)
    rows = []
    for slide_path in tqdm(slides, desc="Deletion curves"):
        try:
            feats = _load_slide_feats(slide_path, zarr_dir, config["feature_key"]).to(device)
        except Exception as e:
            print(f"Skipping {slide_path}: {type(e).__name__}: {e}")
            continue
        rows += [dict(r, slide=_name(slide_path)) for r in _slide_deletion(model, feats, frac_grid, n_random, rng)]
    return pd.DataFrame(rows)


def _area(dele):
    """Per slide and ranking: mean P(non-healthy) over the deleted range (area under curve / range)."""
    wide = dele.pivot_table(index=["slide", "ranking"], columns="frac_deleted", values="prob_pos").sort_index(axis=1)
    x, y = wide.columns.to_numpy(float), wide.to_numpy()
    area = (np.diff(x) * (y[:, 1:] + y[:, :-1]) / 2).sum(axis=1) / (x[-1] - x[0])
    return pd.Series(area, index=wide.index).unstack("ranking")


def table_deletion(dele):
    """
    Per ranking: mean P(non-healthy) over the deleted range (normalised area under the
    curve), its difference from random deletion (paired per slide), and a Wilcoxon
    signed-rank test of that difference. Mean (95% CI) throughout.
    """
    a = _area(dele)
    rows = []
    for k in ["disease", "healthy", "random"]:
        m, lo, hi = _mean_ci(a[k])
        r = {"Deletion order": DELETION[k][0], "Slides": int(a[k].notna().sum()),
             "Mean P(non-healthy), (95% CI)": f"{m:.3f} ({lo:.3f}–{hi:.3f})"}
        if k != "random":
            diff = a[k] - a["random"]
            m, lo, hi = _mean_ci(diff)
            r["Difference vs random (95% CI)"] = f"{m:+.3f} ({lo:+.3f}–{hi:+.3f})"
            r["Wilcoxon p"] = f"{stats.wilcoxon(diff.dropna()).pvalue:.1e}"
        rows.append(r)
    frac = dele["frac_deleted"]
    print(f"Area under P(non-healthy) over {frac.min():.0%}–{frac.max():.0%} of tiles deleted, divided by that range.")
    return pd.DataFrame(rows).fillna("")


def fig_deletion_curves(dele):
    """One panel: mean P(non-healthy) ± 95% CI vs. fraction of tiles deleted, one line per ranking."""
    fig, ax = plt.subplots(figsize=(89 * MM, 65 * MM))
    for k, (label, color) in DELETION.items():
        s = (dele[dele["ranking"] == k].groupby("frac_deleted")["prob_pos"]
             .apply(lambda x: pd.Series(_mean_ci(x), index=["m", "lo", "hi"])).unstack())
        x = 100 * s.index.to_numpy()
        ax.plot(x, s["m"], color=color, lw=1.2, marker="o", ms=2.5, label=label,
                ls="--" if k == "random" else "-")
        ax.fill_between(x, s["lo"], s["hi"], color=color, alpha=0.2, lw=0)
    ax.set_xlabel("Tiles deleted (%)")
    ax.set_ylabel("P(non-healthy), mean ± 95% CI")
    ax.set_ylim(0.2, 1.02)
    ax.set_xlim(0, 100 * dele["frac_deleted"].max())
    ax.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    return cv._done(fig)


# ----------------------------------------
# 3. Model evidence per diagnosis
# ----------------------------------------

def _bias_diff(checkpoint_path):
    """b_non-healthy - b_healthy of the classifier, read without moving the model to GPU."""
    b = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]["classifier.bias"]
    return float(b[1] - b[0])


def evidence_per_slide(cache_path, ext, checkpoint_path):
    """
    Add per-slide evidence measures from the cache to `ext` (from load_predictions).

    With attention a_i summing to 1, the slide logit difference is exactly
        logit_non-healthy - logit_healthy = sum_i a_i * (contrast_i + db),   db = b1 - b0,
    so a tile pushes towards non-healthy when contrast_i + db > 0 (not contrast_i > 0),
    and a_i * (contrast_i + db) are per-tile contributions that sum to the slide logit.

      frac_disease_like:            fraction of tiles with contrast_i + db > 0.
      frac_tiles_half_contribution: smallest fraction of tiles whose positive contributions
                                    a_i * (contrast_i + db) add up to 50% of the slide's total
                                    positive contribution (low = evidence in few tiles).
    """
    db = _bias_diff(checkpoint_path)
    slide_cache = {_name(k): v for k, v in _slide_cache(cache_path).items()}
    rows, err, med_abs = [], [], []
    for slide in ext["slide"]:
        t = slide_cache[slide]["tile_table"]
        shifted = t["contrast_score"].to_numpy() + db
        contrib = t["attention"].to_numpy() * shifted
        err.append(abs(1 / (1 + np.exp(-contrib.sum())) - slide_cache[slide]["prob_1"]))
        med_abs.append(np.median(np.abs(t["contrast_score"])))
        pos = np.sort(contrib.clip(min=0))[::-1]
        n_half = int(np.searchsorted(np.cumsum(pos), 0.5 * pos.sum()) + 1)
        rows.append({"slide": slide, "frac_disease_like": float((shifted > 0).mean()),
                     "frac_tiles_half_contribution": n_half / len(t)})
    print(f"Classifier bias difference db = {db:+.3f} (median |contrast| across slides: {np.median(med_abs):.3f})")
    print(f"Check: |sigmoid(sum of contributions) - P(non-healthy)| max = {max(err):.1e} "
          "(non-zero only if tiles were dropped from the tile table)")
    return ext.merge(pd.DataFrame(rows), on="slide")


def table_evidence_by_diagnosis(evid, group_col, min_n=5):
    """Median [IQR] of each evidence measure per diagnosis (< min_n slides -> 'Other'), Kruskal-Wallis p."""
    g, order = _groups(evid, group_col, min_n)
    rows = []
    for grp in order + ["All"]:
        d = evid if grp == "All" else evid[g == grp]
        r = {group_col: grp, "Slides": len(d)}
        for col, label in EVIDENCE.items():
            q1, med, q3 = d[col].quantile([0.25, 0.5, 0.75])
            r[label.replace("\n", " ")] = f"{med:.3f} [{q1:.3f}–{q3:.3f}]"
        rows.append(r)
    tested = [grp for grp in order if grp != "Other"]  # 'Other' mixes diagnoses, so it is left out of the test
    p_row = {group_col: "Kruskal–Wallis p (excl. Other)", "Slides": ""}
    for col, label in EVIDENCE.items():
        p_row[label.replace("\n", " ")] = (f"{stats.kruskal(*[evid.loc[g == grp, col] for grp in tested]).pvalue:.1e}"
                                           if len(tested) > 1 else "")
    rows.append(p_row)
    return pd.DataFrame(rows)


EVIDENCE_SHORT = {
    "prob_pos": "P(non-healthy)",
    "frac_disease_like": "Disease-like tiles",
    "frac_tiles_half_contribution": "Tiles for 50% of contribution",
}


def _short(label, short_names):
    """Short diagnosis label: short_names[label] if given, else the initials of a long name
    ("Diffuse large B-cell lymphoma" -> "DLBCL", "Follicular lymphoma" -> "FL")."""
    if label in short_names:
        return short_names[label]
    if len(label) <= 6 or label == "Other":
        return label
    return "".join(w[0].upper() for w in re.split(r"[\s\-/,]+", label) if w)


def fig_evidence_by_diagnosis(evid, group_col, min_n=5, short_names=None):
    """
    One panel per evidence measure, box + points per diagnosis. Diagnoses are shown by
    short names (initials unless given in short_names={full: short}); the mapping is
    printed for the figure legend.
    """
    g, order = _groups(evid, group_col, min_n)
    short = {grp: _short(str(grp), short_names or {}) for grp in order}
    if len(set(short.values())) < len(short):
        raise ValueError(f"Short names collide: {short}. Pass short_names to resolve.")
    print("; ".join(f"{s}, {grp}" for grp, s in short.items() if s != grp))
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, len(EVIDENCE), figsize=(183 * MM, 55 * MM))
    for ax, (col, label) in zip(axes, EVIDENCE_SHORT.items()):
        data = [evid.loc[g == grp, col].values for grp in order]
        ax.boxplot(data, positions=range(len(order)), widths=0.5, showfliers=False, medianprops={"color": INK})
        for i, v in enumerate(data):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, alpha=0.6, color=PALETTE[0], lw=0)
        ax.set_xticks(range(len(order)), [f"{short[grp]}\nn={len(v)}" for grp, v in zip(order, data)])
        ax.set_title(label)
        if col != "frac_tiles_half_contribution":  # that one is often near 0; let it autoscale
            ax.set_ylim(-0.02, 1.02)
    axes[0].axhline(0.5, color=MUTED, lw=0.6, ls="--")
    plt.tight_layout()
    return cv._done(fig)
