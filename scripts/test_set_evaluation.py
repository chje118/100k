"""
Paper evaluation of an external test set: one tissue type, one slide per patient,
all slides non-healthy (e.g. lymphoma, endometrium). Scored by one trained ABMIL
checkpoint; uses the saved ABMILInference cache where possible.

    import test_set_evaluation as ev

    # 1. Detection: how many slides are called non-healthy, per diagnosis
    ext = ev.load_predictions(inference_cache, slides, meta, "path", "M_idx", "rekvnr", "diagnose")
    ev.table_detection(ext, "diagnose")
    ev.fig_scores(ext, "diagnose")

    # 2. Faithfulness: deletion curves (attention vs contribution vs random)
    att, con, audc = ev.run_deletion_curves(slides, abmil_path, zarr_dir)
    ev.fig_deletion_curves(att, con, "Lymphoma")

    # 3. Selected tiles: where the cut tiles sit within their slide
    sel = ev.selected_tile_percentiles(inference_cache, slides)   # pass the ROISelector settings used for cutting
    ev.fig_selected_tile_percentiles(sel)

    # 4. Model evidence per diagnosis
    evid = ev.evidence_per_slide(inference_cache, ext)
    ev.table_evidence_by_diagnosis(evid, "diagnose")
    ev.fig_evidence_by_diagnosis(evid, "diagnose")
"""

import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, kruskal, wilcoxon
from tqdm import tqdm

import cv_paper_results as cv
from paper_style import MM, PALETTE, INK, MUTED
from abmil import (load_checkpoint, _load_slide_feats, _slide_deletion_curve,
                   deletion_auc_summary, plot_deletion_curves_comparison)
from roi_selection import ROISelector

SCORES = ["attention", "contrast_score", "contribution_score"]
EVIDENCE = {
    "prob_pos": "P(non-healthy)",
    "frac_disease_like": "Fraction of disease-like tiles",
    "frac_tiles_half_contribution": "Fraction of tiles giving\n50% of contribution",
}


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


# ----------------------------------------
# 1. Detection
# ----------------------------------------

def load_predictions(cache_path, slides, meta, filename_col, label_col, patient_col, group_col):
    """One row per slide (slide, group, patient, label, prob_pos, pred, n_tiles). Reuses cv.from_cache."""
    return cv.from_cache(cache_path, slides, meta, filename_col, label_col, patient_col, group_col)


def table_detection(ext, group_col):
    """
    Slides called non-healthy (threshold 0.5) per diagnosis and overall, with
    exact (Clopper-Pearson) 95% CI. All slides are non-healthy, so this is the
    sensitivity; AUC/specificity are undefined without healthy slides.
    """
    rows = []
    for g, d in list(ext.groupby(group_col)) + [("All", ext)]:
        k, n = int(d["pred"].sum()), len(d)
        ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        rows.append({group_col: g, "Slides": n, "Called non-healthy": k,
                     "Sensitivity (95% CI)": f"{k / n:.3f} ({ci.low:.3f}–{ci.high:.3f})",
                     "Median P(non-healthy)": f"{d['prob_pos'].median():.3f}"})
    return pd.DataFrame(rows)


def fig_scores(ext, group_col):
    """Predicted P(non-healthy) per diagnosis. Reuses cv.fig_scores."""
    order = ext[group_col].value_counts().index.tolist()
    return cv.fig_scores(ext, pd.DataFrame({group_col: order}), group_col)


# ----------------------------------------
# 2. Deletion curves
# ----------------------------------------

def run_deletion_curves(slides, checkpoint_path, zarr_dir, frac_grid=None, n_random=5, seed=0):
    """
    Deletion curves with one checkpoint, ranked by attention and by contribution
    score, each vs. random deletion. Reuses abmil._slide_deletion_curve, so the
    output has the same columns as run_deletion_curve_evaluation.

    Returns (results_attention, results_contribution, audc_table). audc_table has
    the mean AUDC per ranking and a paired Wilcoxon test of AUDC_random - AUDC_ranking.
    """
    if frac_grid is None:
        frac_grid = np.linspace(0.0, 1.0, 11)
    model, config, _ = load_checkpoint(checkpoint_path)
    device = next(model.parameters()).device
    feature_key = config["feature_key"]

    results = {}
    for rank_by in ["attention", "contribution_score"]:
        rng = np.random.RandomState(seed)
        rows = []
        for slide_path in tqdm(slides, desc=f"Deletion curves ({rank_by})"):
            try:
                feats = _load_slide_feats(slide_path, zarr_dir, feature_key)
            except Exception as e:
                print(f"Skipping {slide_path}: {type(e).__name__}: {e}")
                continue
            r = _slide_deletion_curve(model, feats, frac_grid, device, n_random=n_random, rng=rng, rank_by=rank_by)
            for frac, imp_prob, rand_prob in zip(frac_grid, r["importance_curve"], r["random_curve"]):
                rows.append({"slide_path": slide_path, "pred_idx": r["pred_idx"], "rank_by": rank_by,
                             "baseline_prob": r["baseline_prob"], "frac_deleted": frac,
                             "importance_prob": imp_prob, "random_prob": rand_prob})
        results[rank_by] = pd.DataFrame(rows)

    audc_rows = []
    for rank_by, df in results.items():
        audc = deletion_auc_summary(df)
        audc_rows.append({"Ranking": rank_by, "Slides": len(audc),
                          "AUDC ranking": f"{audc['audc_importance'].mean():.3f} ± {audc['audc_importance'].std():.3f}",
                          "AUDC random": f"{audc['audc_random'].mean():.3f} ± {audc['audc_random'].std():.3f}",
                          "Gap (random - ranking)": f"{audc['audc_gap'].mean():.3f} ± {audc['audc_gap'].std():.3f}",
                          "Wilcoxon p": f"{wilcoxon(audc['audc_gap']).pvalue:.1e}"})
    return results["attention"], results["contribution_score"], pd.DataFrame(audc_rows)


def fig_deletion_curves(results_attention, results_contribution, name):
    """Attention, contribution and random deletion in one plot. Reuses plot_deletion_curves_comparison."""
    return plot_deletion_curves_comparison({name: (results_attention, results_contribution)},
                                           title=f"Deletion curves - {name}")


# ----------------------------------------
# 3. Selected tiles
# ----------------------------------------

def selected_tile_percentiles(cache_path, slides, **selector_kwargs):
    """
    Run ROISelector on every slide (pass the settings used for cutting) and return
    one row per selected tile with its scores and within-slide percentile
    (0 = lowest tile in the slide, 100 = highest). Prints a median [IQR] summary
    and the number of tiles that could not be selected.
    """
    slide_cache = _slide_cache(cache_path)

    class _Selector(ROISelector):  # reuse the loaded cache instead of re-reading it per slide
        @staticmethod
        def load_cache(_):
            return slide_cache

    rows, shortfall = [], []
    for slide_path in tqdm(slides, desc="Selecting tiles"):
        if slide_path not in slide_cache:
            print(f"Skipping {slide_path}: not in inference cache")
            continue
        selector = _Selector(cache_path=cache_path, slide_path=slide_path, **selector_kwargs)
        disease_tiles, healthy_tiles = selector.select_tiles()

        tiles = slide_cache[slide_path]["tile_table"].dropna(subset=SCORES)
        pct = tiles[SCORES].rank(pct=True).mul(100).add_suffix("_pct")
        tiles = pd.concat([tiles[["tile_id"] + SCORES], pct], axis=1)

        for arm, selected, k in [("disease", disease_tiles, selector.disease_k),
                                 ("healthy", healthy_tiles, selector.healthy_k)]:
            rows.append(tiles[tiles["tile_id"].isin(selected["tile_id"])].assign(slide_path=slide_path, arm=arm))
            shortfall.append({"arm": arm, "missing": k - len(selected)})

    sel_df = pd.concat(rows, ignore_index=True)
    missing = pd.DataFrame(shortfall).groupby("arm")["missing"].agg(total="sum", slides=lambda x: (x > 0).sum())
    print(f"{sel_df['slide_path'].nunique()} slides")
    for arm, m in missing.iterrows():
        print(f"  {arm} arm: {(sel_df['arm'] == arm).sum()} tiles selected, "
              f"{m['total']} not assigned (in {m['slides']} slides)")
    for c in SCORES:
        for arm, v in sel_df.groupby("arm")[f"{c}_pct"]:
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            print(f"  {c:<20} {arm:<8} median percentile {med:.1f} [IQR {q1:.1f}-{q3:.1f}]")
    return sel_df


def fig_selected_tile_percentiles(sel_df):
    """One panel per score: within-slide percentile of the selected tiles, per arm."""
    labels = {"attention": "Attention", "contrast_score": "Contrast score", "contribution_score": "Contribution score"}
    colors = {"disease": PALETTE[1], "healthy": PALETTE[2]}
    arms = ["disease", "healthy"]
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, len(SCORES), figsize=(183 * MM, 60 * MM), sharey=True)
    for ax, score in zip(axes, SCORES):
        data = [sel_df.loc[sel_df["arm"] == arm, f"{score}_pct"].values for arm in arms]
        ax.boxplot(data, positions=range(len(arms)), widths=0.5, showfliers=False, medianprops={"color": INK})
        for i, (arm, v) in enumerate(zip(arms, data)):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=3, alpha=0.4, color=colors[arm], lw=0)
        ax.axhline(50, color=MUTED, lw=0.6, ls="--")
        ax.set_xticks(range(len(arms)), [f"{a.capitalize()} arm" for a in arms])
        ax.set_title(labels[score])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Within-slide percentile")
    plt.tight_layout()
    return cv._done(fig)


# ----------------------------------------
# 4. Model evidence per diagnosis
# ----------------------------------------

def evidence_per_slide(cache_path, ext):
    """
    Add per-slide evidence measures from the cache to `ext` (from load_predictions):
      frac_disease_like:            fraction of tiles with contrast_score > 0.
      frac_tiles_half_contribution: smallest fraction of tiles whose positive
                                    contributions add up to 50% of the slide's total
                                    positive contribution (low = evidence concentrated
                                    in few tiles).
    """
    slide_cache = {_name(k): v for k, v in _slide_cache(cache_path).items()}
    rows = []
    for slide in ext["slide"]:
        t = slide_cache[slide]["tile_table"]
        pos = np.sort(t["contribution_score"].clip(lower=0).to_numpy())[::-1]
        n_half = int(np.searchsorted(np.cumsum(pos), 0.5 * pos.sum()) + 1)
        rows.append({"slide": slide, "frac_disease_like": float((t["contrast_score"] > 0).mean()),
                     "frac_tiles_half_contribution": n_half / len(t)})
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
        p_row[label.replace("\n", " ")] = (f"{kruskal(*[evid.loc[g == grp, col] for grp in tested]).pvalue:.1e}"
                                           if len(tested) > 1 else "")
    rows.append(p_row)
    return pd.DataFrame(rows)


def fig_evidence_by_diagnosis(evid, group_col, min_n=5):
    """One panel per evidence measure, box + points per diagnosis."""
    g, order = _groups(evid, group_col, min_n)
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, len(EVIDENCE), figsize=(183 * MM, 65 * MM))
    for ax, (col, label) in zip(axes, EVIDENCE.items()):
        data = [evid.loc[g == grp, col].values for grp in order]
        ax.boxplot(data, positions=range(len(order)), widths=0.5, showfliers=False, medianprops={"color": INK})
        for i, v in enumerate(data):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, alpha=0.6, color=PALETTE[0], lw=0)
        ax.set_xticks(range(len(order)), [f"{grp} (n={len(v)})" for grp, v in zip(order, data)],
                      rotation=30, ha="right")
        ax.set_ylabel(label)
    plt.tight_layout()
    return cv._done(fig)
