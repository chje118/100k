"""
Paper tables, figures and numbers for the 5-fold CV across specified groups.
Every function RETURNS a DataFrame, a matplotlib Figure or a string.

Input: `oof` from oof_group_metrics.collect_oof_predictions
(columns: slide, fold, <group>, <patient>, label (0/1), prob_pos, pred, n_tiles).

    import cv_paper_results as cv

    t = cv.organ_table(oof, "tissue", "rekvnr")              # all numbers per organ (+ 'All organs')

    # Main
    cv.table_cohort(oof, "tissue", "rekvnr")                 # Table 1
    cv.table_per_organ(t, "tissue")                          # Table 2
    cv.fig_cv_roc(oof, t, "tissue", groups=["Lymphoma"])     # Fig. 1 (groups: optional extra organ curves)
    cv.fig_forest(t, "tissue")                               # Fig. 2

    # Supplement
    cv.table_per_fold(oof, "rekvnr")
    cv.table_organ_by_fold(oof, "tissue")
    cv.fig_roc_per_organ(oof, t, "tissue")
    cv.fig_scores(oof, t, "tissue")
    cv.fig_confusion(oof, "tissue", "Lymphoma")              # one organ per call
    cv.fig_confusion(oof, "tissue", cv.ALL)                  # all organs pooled

    # Numbers + draft Methods / Results / legends
    print(cv.paper_text(oof, t, "tissue", configs=oof.attrs.get("configs"), skipped=oof.attrs.get("skipped")))

    # External cohort (final model, no folds): same table from the ABMILInference cache, then the same calls
    ext = cv.from_cache(cache_path, slides, meta, "filename", "label", "rekvnr", "tissue")  # group optional
    t = cv.organ_table(ext, "tissue", "rekvnr")
    cv.table_cohort(ext, "tissue", "rekvnr"); cv.table_per_organ(t, "tissue"); cv.fig_cv_roc(ext, t, "tissue")
    cv.fig_forest(t, "tissue"); cv.fig_scores(ext, t, "tissue"); cv.fig_confusion(ext, "tissue", cv.ALL)

Statistics
- Each slide is scored once, by the fold model that never saw it (out-of-fold, OOF).
- Threshold fixed at 0.5 for every organ (not tuned).
- 95% CI: percentile bootstrap resampling PATIENTS (slides of one patient stay together), 2000 resamples.
- Fold mean ± SD uses np.std (ddof=0), as in KFoldPipeline, so it matches the training log.
- Within-organ AUC = slide-weighted mean of per-organ AUCs; if clearly below the pooled AUC, part of
  the pooled AUC comes from organs differing in prevalence/score level, not discrimination within an organ.
- Tile-count baseline (if `n_tiles` is present): AUC of ranking slides by number of tiles alone.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, roc_curve

ALL = "All organs"
from paper_style import MM, INK, INK2, MUTED, MAIN_C, POS_C, NEG_C, BASE_C, EXTRA_C  # shared figure style


# ----------------------------------------
# Metrics
# ----------------------------------------
CI_METRICS = ["auc", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv"]


def metrics(y, p, pred):
    y, p, pred = np.asarray(y), np.asarray(p), np.asarray(pred)
    tp, fn = np.sum((pred == 1) & (y == 1)), np.sum((pred == 0) & (y == 1))
    tn, fp = np.sum((pred == 0) & (y == 0)), np.sum((pred == 1) & (y == 0))
    div = lambda a, b: a / b if b else np.nan
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    return {"auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "accuracy": np.mean(y == pred), "balanced_accuracy": np.nanmean([sens, spec]),
            "sensitivity": sens, "specificity": spec, "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def bootstrap(d, patient_col, n_boot, rng, extra_score=None):
    """Patient-level percentile 95% CI for CI_METRICS (+ AUC of `extra_score` and the AUC difference)."""
    y, p, pred = d["label"].to_numpy(), d["prob_pos"].to_numpy(), d["pred"].to_numpy()
    s = None if extra_score is None else d[extra_score].to_numpy()
    idx = list(d.groupby(patient_col).indices.values())
    draws = {k: [] for k in CI_METRICS + (["base_auc", "auc_diff"] if s is not None else [])}
    for _ in range(n_boot):
        ix = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        m = metrics(y[ix], p[ix], pred[ix])
        for k in CI_METRICS:
            draws[k].append(m[k])
        if s is not None:
            b = roc_auc_score(y[ix], s[ix]) if len(np.unique(y[ix])) == 2 else np.nan
            draws["base_auc"].append(b)
            draws["auc_diff"].append(m["auc"] - b)
    out = {}
    for k, v in draws.items():
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        # CI only when most resamples are valid (small organs can draw a single class)
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(v, [2.5, 97.5]) if len(v) >= 0.9 * n_boot else (np.nan, np.nan)
        if k == "auc_diff":
            out["auc_diff_p"] = max(float((v <= 0).mean()), 1 / len(v)) if len(v) else np.nan  # one-sided
    return out


def organ_table(oof, group_col, patient_col, n_boot=2000, seed=0):
    """All numbers, one row per organ (sorted by AUC) + 'All organs' (pooled). Input for the other functions."""
    rng = np.random.default_rng(seed)
    base = "n_tiles" if "n_tiles" in oof.columns else None
    rows = []
    for g, d in list(oof.groupby(group_col)) + [(ALL, oof)]:
        y = d["label"].to_numpy()
        r = {group_col: g, "n_slides": len(d), "n_patients": d[patient_col].nunique(),
             "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "prevalence": y.mean(),
             **metrics(y, d["prob_pos"], d["pred"]), **bootstrap(d, patient_col, n_boot, rng, base)}
        if base:
            r["base_auc"] = roc_auc_score(y, d[base]) if len(np.unique(y)) == 2 else np.nan
            r["auc_diff"] = r["auc"] - r["base_auc"]
        rows.append(r)
    t = pd.DataFrame(rows)
    body = t[t[group_col] != ALL].sort_values("auc", ascending=False)
    return pd.concat([body, t[t[group_col] == ALL]], ignore_index=True)


def from_cache(cache_path, slides, meta, filename_col, label_col, patient_col, group_col=None):
    """External cohort: predictions from an ABMILInference cache (.pkl), as an OOF-like table without `fold`
    (slide, group, patient, label, prob_pos, pred, n_tiles). Only slides in `slides` are used; every other cached
    slide is ignored. Matched on meta[filename_col] by file name (path and extension ignored), so it can also be
    an id the scans are named by, e.g. table_id. group_col=None: one group, column "group" = "All slides"."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    name = lambda p: os.path.splitext(os.path.basename(str(p)))[0]  # "D:/x/a.mrxs", "a.mrxs" and "a" all match
    cache = {name(k): v for k, v in cache.get("slide_cache", cache).items()}
    if len(set(map(name, slides))) < len(slides):
        raise ValueError("Duplicate names in `slides`")
    print(f"Cache: {len(cache)} slides; using {sum(name(s) in cache for s in slides)} from `slides`, "
          f"ignoring {len(set(cache) - set(map(name, slides)))}")
    missing = [s for s in slides if name(s) not in cache]
    if missing:
        print(f"Not in cache, left out ({len(missing)}): {missing}")
    rows = [{"slide": name(s), "prob_pos": cache[name(s)]["prob_1"], "pred": int(cache[name(s)]["pred_idx"]),
             "n_tiles": len(cache[name(s)]["attention"])} for s in slides if name(s) in cache]
    if group_col is None:
        meta, group_col = meta.assign(group="All slides"), "group"
    m = meta.assign(slide=meta[filename_col].map(name))[["slide", group_col, patient_col, label_col]]
    if m["slide"].duplicated().any():
        raise ValueError(f"Duplicate '{filename_col}' in meta: {m.loc[m['slide'].duplicated(), 'slide'].tolist()[:5]}")
    d = pd.DataFrame(rows).merge(m.rename(columns={label_col: "label"}), on="slide", how="left")
    if d["label"].isna().any():
        raise ValueError(f"No '{label_col}' in meta for: {d.loc[d['label'].isna(), 'slide'].tolist()}")
    d["label"] = d["label"].astype(int)
    print(f"{len(d)} slides, {d[patient_col].nunique()} patients, {d['label'].sum()} positive / {(1 - d['label']).sum()} negative")
    return d


def within_organ_auc(t, group_col):
    o = t[(t[group_col] != ALL) & t["auc"].notna()]
    return np.average(o["auc"], weights=o["n_slides"]) if len(o) else np.nan


def heterogeneity(t, group_col):
    """Cochran's Q / I² for per-organ AUCs, SE = CI width / 3.92 (patients belong to one organ)."""
    o = t[t[group_col] != ALL].dropna(subset=["auc_lo", "auc_hi"])
    se = (o["auc_hi"] - o["auc_lo"]).to_numpy(float) / 3.92
    o, se = o[se > 0], se[se > 0]
    if len(o) < 2:
        return {"Q": np.nan, "df": 0, "p": np.nan, "I2": np.nan}
    w = 1 / se ** 2
    q = float(np.sum(w * (o["auc"] - np.sum(w * o["auc"]) / np.sum(w)) ** 2))
    dof = len(o) - 1
    return {"Q": q, "df": dof, "p": float(chi2.sf(q, dof)), "I2": max(0.0, (q - dof) / q) if q > 0 else 0.0}


def fold_aucs(oof):
    return pd.Series({k: roc_auc_score(f["label"], f["prob_pos"]) for k, f in oof.groupby("fold")})


# ----------------------------------------
# Tables
# ----------------------------------------
def _ci(r, m, d=3):
    if pd.isna(r[m]):
        return "n/a"
    lo, hi = r.get(f"{m}_lo", np.nan), r.get(f"{m}_hi", np.nan)
    return f"{r[m]:.{d}f}" if pd.isna(lo) else f"{r[m]:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def table_per_organ(t, group_col):
    """Table 2: per-organ performance with 95% CI, plus the within-organ AUC."""
    rows = [{"Organ": r[group_col], "Slides": int(r["n_slides"]), "Patients": int(r["n_patients"]),
             "Pos / Neg": f"{int(r['n_pos'])} / {int(r['n_neg'])}",
             "AUC (95% CI)": _ci(r, "auc"), "Balanced acc. (95% CI)": _ci(r, "balanced_accuracy"),
             "Sensitivity (95% CI)": _ci(r, "sensitivity"), "Specificity (95% CI)": _ci(r, "specificity"),
             "PPV (95% CI)": _ci(r, "ppv"), "NPV (95% CI)": _ci(r, "npv")} for _, r in t.iterrows()]
    rows.append({"Organ": "Within-organ AUC (slide-weighted mean)",
                 "AUC (95% CI)": f"{within_organ_auc(t, group_col):.3f}"})
    return pd.DataFrame(rows).fillna("")


def table_cohort(oof, group_col, patient_col, pos_name="Non-healthy", neg_name="Healthy"):
    """Table 1: slides, patients and class counts per organ."""
    rows = []
    for g, d in list(oof.groupby(group_col)) + [(ALL, oof)]:
        spp = d.groupby(patient_col).size()
        rows.append({"Organ": g, "Slides": len(d), "Patients": d[patient_col].nunique(),
                     pos_name: int(d["label"].sum()), neg_name: int((1 - d["label"]).sum()),
                     f"% {pos_name}": f"{d['label'].mean():.1%}",
                     "Slides per patient, median (range)": f"{int(spp.median())} ({spp.min()}–{spp.max()})"})
    return pd.DataFrame(rows)


def table_per_fold(oof, patient_col):
    """Supplementary: size and performance of each test fold, plus mean ± SD."""
    num = {"auc": "AUC", "balanced_accuracy": "Balanced acc.", "accuracy": "Accuracy",
           "sensitivity": "Sensitivity", "specificity": "Specificity"}
    rows = []
    for k, f in oof.groupby("fold"):
        m = metrics(f["label"], f["prob_pos"], f["pred"])
        rows.append({"Fold": int(k), "Slides": len(f), "Patients": f[patient_col].nunique(),
                     "Pos / Neg": f"{int(f['label'].sum())} / {int((1 - f['label']).sum())}",
                     **{n: m[c] for c, n in num.items()}})
    raw = pd.DataFrame(rows)
    out = raw.copy()
    for c in num.values():
        out[c] = out[c].map(lambda v: f"{v:.3f}")
    out.loc[len(out)] = {"Fold": "Mean ± SD", "Slides": int(raw["Slides"].sum()),
                         "Patients": oof[patient_col].nunique(),
                         "Pos / Neg": f"{int(oof['label'].sum())} / {int((1 - oof['label']).sum())}",
                         **{c: f"{raw[c].mean():.3f} ± {raw[c].std(ddof=0):.3f}" for c in num.values()}}
    return out


def table_organ_by_fold(oof, group_col):
    """Supplementary: slides per organ in each test fold."""
    tab = pd.crosstab(oof[group_col], oof["fold"], margins=True, margins_name="Total")
    tab.columns = [c if c == "Total" else f"Fold {c}" for c in tab.columns]
    return tab


# ----------------------------------------
# Figures (each returns a Figure; save yourself with fig.savefig(...) if wanted)
# ----------------------------------------
def _done(fig):
    plt.close(fig)  # avoid showing twice in Jupyter; the returned Figure still displays
    return fig


def _organs(t, group_col):
    return t[t[group_col] != ALL][group_col].tolist()


def _small_multiples(n, ncols, w_mm, h_mm, **kw):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w_mm * MM, h_mm * MM * nrows), **kw)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes, nrows


def fig_cv_roc(oof, t, group_col, groups=None):
    """Fig. 1: ROC of each test fold (thin), pooled OOF (thick), tile-count baseline (dashed), random (dotted).
    groups: optional list of organs to add as their own curve, e.g. ["Lymphoma"]."""
    allr = t[t[group_col] == ALL].iloc[0]
    has_folds = "fold" in oof.columns  # external cohort: no folds, single curve
    fa = fold_aucs(oof) if has_folds else None
    groups = [] if groups is None else [groups] if isinstance(groups, str) else list(groups)
    unknown = set(groups) - set(oof[group_col])
    if unknown:
        raise ValueError(f"Not in '{group_col}': {sorted(unknown)}. Options: {sorted(oof[group_col].unique())}")
    fig, ax = plt.subplots(figsize=(89 * MM, 89 * MM))
    for i, (_, f) in enumerate(oof.groupby("fold") if has_folds else []):
        fpr, tpr, _ = roc_curve(f["label"], f["prob_pos"])
        ax.plot(fpr, tpr, color=MAIN_C, lw=0.7, alpha=0.35,
                label=f"Test folds (k={len(fa)}), AUC {fa.mean():.3f} ± {fa.std(ddof=0):.3f}" if i == 0 else None)
    fpr, tpr, _ = roc_curve(oof["label"], oof["prob_pos"])
    ax.plot(fpr, tpr, color=MAIN_C, lw=1.8, label=f"{'Pooled' if has_folds else 'All slides'}, AUC {_ci(allr, 'auc')}")
    ax.plot(1 - allr["specificity"], allr["sensitivity"], "o", ms=4, color=MAIN_C, mec="white", mew=0.6)
    if "base_auc" in t.columns:
        fpr, tpr, _ = roc_curve(oof["label"], oof["n_tiles"])
        ax.plot(fpr, tpr, color=BASE_C, lw=1.2, ls=(0, (4, 2)), label=f"Tile count only, AUC {_ci(allr, 'base_auc')}")
    for g, c in zip(groups, EXTRA_C):
        d, r = oof[oof[group_col] == g], t[t[group_col] == g].iloc[0]
        if d["label"].nunique() < 2:
            print(f"Skipping '{g}' in the ROC: only one class")
            continue
        fpr, tpr, _ = roc_curve(d["label"], d["prob_pos"])
        ax.plot(fpr, tpr, color=c, lw=1.4, label=f"{g}, AUC {_ci(r, 'auc')}")
        print(f"{g}: n = {len(d)} slides ({int(r['n_pos'])} positive / {int(r['n_neg'])} negative), "
              f"{int(r['n_patients'])} patients")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":", label="Random, AUC 0.500")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 − Specificity", ylabel="Sensitivity", aspect="equal")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=1)
    return _done(fig)


def fig_forest(t, group_col):
    """Fig. 2: AUC, sensitivity, specificity per organ with 95% CI."""
    o = t[t[group_col] != ALL].reset_index(drop=True)
    allr = t[t[group_col] == ALL].iloc[0]
    labels = [f"{g}  (n={n}; {p}/{q})" for g, n, p, q in zip(o[group_col], o["n_slides"], o["n_pos"], o["n_neg"])]
    yy = np.arange(len(o))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, (25 + 6 * len(o)) * MM), sharey=True)
    for ax, (m, title) in zip(axes, [("auc", "AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity")]):
        v, lo, hi = (o[c].to_numpy(float) for c in (m, f"{m}_lo", f"{m}_hi"))
        ok = ~np.isnan(v)
        err = np.nan_to_num(np.vstack([v - lo, hi - v]))
        ax.errorbar(v[ok], yy[ok], xerr=err[:, ok], fmt="o", ms=4, color=MAIN_C, elinewidth=1, capsize=0,
                    mec="white", mew=0.6, zorder=3)
        for yi in yy[~ok]:
            ax.text(0.52, yi, "n/a (one class)", va="center", fontsize=5.5, color=MUTED)
        ax.axvline(allr[m], color=INK2, lw=0.8, ls="--", zorder=1)
        ax.text(allr[m], len(o) - 0.15, f"all {allr[m]:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK2,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        if m == "auc":
            ax.axvline(0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.set_xlim(0.45 if m == "auc" else 0, 1.02)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_yticks(yy, labels)
    axes[0].set_ylim(-0.7, len(o) + 0.3)
    return _done(fig)


def fig_roc_per_organ(oof, t, group_col, ncols=3):
    """Supplementary: ROC per organ, pooled ROC in grey, dot = threshold 0.5."""
    organs = _organs(t, group_col)
    fpr_all, tpr_all, _ = roc_curve(oof["label"], oof["prob_pos"])
    fig, axes, nrows = _small_multiples(len(organs), ncols, 183, 58, sharex=True, sharey=True)
    for ax, g in zip(axes, organs):
        d, r = oof[oof[group_col] == g], t[t[group_col] == g].iloc[0]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":")
        ax.plot(fpr_all, tpr_all, color="#c3c2b7", lw=1.2, label=ALL)
        if d["label"].nunique() == 2:
            fpr, tpr, _ = roc_curve(d["label"], d["prob_pos"])
            ax.plot(fpr, tpr, color=MAIN_C, lw=1.5, label="This organ")
            ax.plot(1 - r["specificity"], r["sensitivity"], "o", ms=4, color=MAIN_C, mec="white", mew=0.6)
        ax.text(0.97, 0.05, f"AUC {_ci(r, 'auc')}\nn={len(d)} ({int(r['n_pos'])}/{int(r['n_neg'])})",
                ha="right", va="bottom", fontsize=5.5, color=INK)
        ax.set_title(g, loc="left", fontweight="bold")
        ax.set_aspect("equal")
    for ax in axes[::ncols]:
        ax.set_ylabel("Sensitivity")
    for ax in axes[(nrows - 1) * ncols:]:
        ax.set_xlabel("1 − Specificity")
    axes[0].legend(loc="center right", frameon=False, handlelength=1.2)
    return _done(fig)


def fig_scores(oof, t, group_col, pos_name="Non-healthy", neg_name="Healthy"):
    """Supplementary: predicted P(positive) per organ by true class (median + IQR)."""
    organs = _organs(t, group_col)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(183 * MM, 65 * MM))
    for i, g in enumerate(organs):
        d = oof[oof[group_col] == g]
        for cls, dx, c in [(0, -0.18, NEG_C), (1, 0.18, POS_C)]:
            v = d.loc[d["label"] == cls, "prob_pos"].to_numpy()
            if not len(v):
                continue
            ax.scatter(i + dx + rng.uniform(-0.1, 0.1, len(v)), v, s=3, color=c,
                       alpha=0.35 if len(v) > 200 else 0.6, lw=0, zorder=2)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i + dx] * 2, [q1, q3], color=INK, lw=1.2, zorder=3)
            ax.plot([i + dx - 0.1, i + dx + 0.1], [med] * 2, color=INK, lw=1.5, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks(range(len(organs)), organs, rotation=30, ha="right")
    ax.set_ylabel(f"Predicted P({pos_name})")
    ax.set_ylim(-0.02, 1.02)
    ax.scatter([], [], s=12, color=NEG_C, label=f"True {neg_name}")
    ax.scatter([], [], s=12, color=POS_C, label=f"True {pos_name}")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    return _done(fig)


def fig_confusion(oof, group_col, group=ALL, pos_name="Non-healthy", neg_name="Healthy"):
    """Standard confusion matrix (rows = true label, columns = predicted label, counts) for one organ
    or ALL (pooled), threshold 0.5. Call once per organ, e.g. fig_confusion(oof, "tissue", "Lymphoma")."""
    if group == ALL:
        d = oof
    elif group in set(oof[group_col]):
        d = oof[oof[group_col] == group]
    else:
        raise ValueError(f"'{group}' not in '{group_col}'. Options: {[ALL] + sorted(oof[group_col].unique())}")
    cm = np.array([[np.sum((d["label"] == a) & (d["pred"] == b)) for b in (0, 1)] for a in (0, 1)])
    classes = [neg_name, pos_name]
    fig, ax = plt.subplots(figsize=(60 * MM, 55 * MM))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for a in (0, 1):
        for b in (0, 1):
            ax.text(b, a, f"{cm[a, b]}", ha="center", va="center", fontsize=8,
                    color="white" if cm[a, b] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], classes)
    ax.set_yticks([0, 1], classes, rotation=90, va="center")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{group} (n={len(d)})", fontsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return _done(fig)