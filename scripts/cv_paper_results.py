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
    cv.fig_cv_roc(oof, t, "tissue")                          # Fig. 1
    cv.fig_forest(t, "tissue")                               # Fig. 2

    # Supplement
    cv.table_per_fold(oof, "rekvnr")
    cv.table_organ_by_fold(oof, "tissue")
    cv.fig_roc_per_organ(oof, t, "tissue")
    cv.fig_scores(oof, t, "tissue")
    cv.fig_confusion(oof, t, "tissue")

    # Numbers + draft Methods / Results / legends
    print(cv.paper_text(oof, t, "tissue", configs=oof.attrs.get("configs"), skipped=oof.attrs.get("skipped")))

Statistics
- Each slide is scored once, by the fold model that never saw it (out-of-fold, OOF).
- Threshold fixed at 0.5 for every organ (not tuned).
- 95% CI: percentile bootstrap resampling PATIENTS (slides of one patient stay together), 2000 resamples.
- Fold mean ± SD uses np.std (ddof=0), as in KFoldPipeline, so it matches the training log.
- Within-organ AUC = slide-weighted mean of per-organ AUCs; if clearly below the pooled AUC, part of
  the pooled AUC comes from organs differing in prevalence/score level, not discrimination within an organ.
- Tile-count baseline (if `n_tiles` is present): AUC of ranking slides by number of tiles alone.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, roc_curve

ALL = "All organs"
MM = 1 / 25.4  # Nature: 89 mm single, 183 mm double column
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
MAIN_C, POS_C, NEG_C, BASE_C = "#2a78d6", "#eb6834", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150,
})


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


def _grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _organs(t, group_col):
    return t[t[group_col] != ALL][group_col].tolist()


def _small_multiples(n, ncols, w_mm, h_mm, **kw):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w_mm * MM, h_mm * MM * nrows), **kw)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes, nrows


def fig_cv_roc(oof, t, group_col):
    """Fig. 1: ROC of each test fold (thin), pooled OOF (thick), tile-count baseline (dashed)."""
    allr = t[t[group_col] == ALL].iloc[0]
    fa = fold_aucs(oof)
    fig, ax = plt.subplots(figsize=(89 * MM, 89 * MM))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":")
    for i, (_, f) in enumerate(oof.groupby("fold")):
        fpr, tpr, _ = roc_curve(f["label"], f["prob_pos"])
        ax.plot(fpr, tpr, color=MAIN_C, lw=0.7, alpha=0.35,
                label=f"Test folds (k={len(fa)}), AUC {fa.mean():.3f} ± {fa.std(ddof=0):.3f}" if i == 0 else None)
    fpr, tpr, _ = roc_curve(oof["label"], oof["prob_pos"])
    ax.plot(fpr, tpr, color=MAIN_C, lw=1.8, label=f"Pooled, AUC {_ci(allr, 'auc')}")
    ax.plot(1 - allr["specificity"], allr["sensitivity"], "o", ms=4, color=MAIN_C, mec="white", mew=0.6)
    if "base_auc" in t.columns:
        fpr, tpr, _ = roc_curve(oof["label"], oof["n_tiles"])
        ax.plot(fpr, tpr, color=BASE_C, lw=1.2, ls=(0, (4, 2)), label=f"Tile count only, AUC {_ci(allr, 'base_auc')}")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 − Specificity", ylabel="Sensitivity", aspect="equal")
    ax.legend(frameon=False, loc="lower right")
    _grid(ax)
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
        _grid(ax, "x")
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
        _grid(ax)
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
    _grid(ax, "y")
    return _done(fig)


def fig_confusion(oof, t, group_col, pos_name="Non-healthy", neg_name="Healthy", ncols=3):
    """Supplementary: confusion matrix per organ and pooled (counts + row %), threshold 0.5."""
    organs = _organs(t, group_col) + [ALL]
    fig, axes, _ = _small_multiples(len(organs), ncols, 150, 50, gridspec_kw={"hspace": 0.55, "wspace": 0.45})
    for ax, g in zip(axes, organs):
        d = oof if g == ALL else oof[oof[group_col] == g]
        cm = np.array([[np.sum((d["label"] == a) & (d["pred"] == b)) for b in (0, 1)] for a in (0, 1)])
        rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for a in (0, 1):
            for b in (0, 1):
                ax.text(b, a, f"{cm[a, b]}\n{rown[a, b]:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if rown[a, b] > 0.6 else INK)
        ax.set_xticks([0, 1], [neg_name, pos_name])
        ax.set_yticks([0, 1], [neg_name, pos_name])
        ax.set_title(g, loc="left", fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.supxlabel("Predicted", fontsize=7)
    fig.supylabel("True", fontsize=7)
    return _done(fig)


# ----------------------------------------
# Text
# ----------------------------------------
def paper_text(oof, t, group_col, pos_name="Non-healthy", neg_name="Healthy", n_boot=2000,
               configs=None, skipped=None):
    """Text (print it): key numbers, draft Methods (statistics), Results and legends. Check every sentence.
    configs / skipped: oof.attrs['configs'] / oof.attrs['skipped'] (optional; lost if oof was read from CSV)."""
    allr = t[t[group_col] == ALL].iloc[0]
    o = t[t[group_col] != ALL]
    oa = o.dropna(subset=["auc"])
    best, worst = oa.loc[oa["auc"].idxmax()], oa.loc[oa["auc"].idxmin()]
    within = within_organ_auc(t, group_col)
    diff = allr["auc"] - within
    het = heterogeneity(t, group_col)
    fa = fold_aucs(oof)
    k = oof["fold"].nunique()
    cfg = next(iter((configs or {}).values()), {})
    skipped = pd.DataFrame() if skipped is None else skipped
    n_skip = len(skipped)
    one_class = o[o["auc"].isna()][group_col].tolist()
    has_base = "base_auc" in t.columns

    numbers = [
        f"- Cohort: {int(allr['n_slides'])} slides, {int(allr['n_patients'])} patients, {len(o)} organs; "
        f"{int(allr['n_pos'])} {pos_name} / {int(allr['n_neg'])} {neg_name}.",
        f"- Unreadable at evaluation, excluded: {n_skip} slides"
        + (f" ({skipped[group_col].value_counts().to_dict()})." if n_skip else "."),
        f"- {k}-fold AUC, mean ± SD over folds: {fa.mean():.3f} ± {fa.std(ddof=0):.3f} (range {fa.min():.3f}–{fa.max():.3f}).",
        f"- Pooled OOF AUC {_ci(allr, 'auc')}; sensitivity {_ci(allr, 'sensitivity')}; "
        f"specificity {_ci(allr, 'specificity')}; balanced accuracy {_ci(allr, 'balanced_accuracy')}; "
        f"PPV {allr['ppv']:.3f}; NPV {allr['npv']:.3f} (TP {allr['tp']}, FP {allr['fp']}, TN {allr['tn']}, FN {allr['fn']}).",
        f"- Per-organ AUC {worst['auc']:.3f} ({worst[group_col]}) to {best['auc']:.3f} ({best[group_col]}); "
        f"{int((oa['auc'] >= 0.9).sum())}/{len(oa)} organs ≥ 0.90.",
        f"- Within-organ AUC (slide-weighted): {within:.3f} (pooled − within {diff:+.3f}).",
        f"- Heterogeneity across organs: Q = {het['Q']:.1f}, df = {het['df']}, p = {het['p']:.2g}, I² = {het['I2']:.0%}.",
    ]
    if one_class:
        numbers.append(f"- AUC undefined (one class only): {', '.join(map(str, one_class))}.")
    if has_base:
        numbers.append(f"- Tile count alone: AUC {_ci(allr, 'base_auc')}; classifier − tile count "
                       f"{_ci(allr, 'auc_diff')}, one-sided bootstrap p {'<' if allr['auc_diff_p'] <= 1 / n_boot else '='} "
                       f"{allr['auc_diff_p']:.3g}.")
    numbers += ["", "Per organ:"] + [
        f"  - {r[group_col]}: AUC {_ci(r, 'auc')}, n = {r['n_slides']} ({r['n_pos']}/{r['n_neg']}), "
        f"sens {r['sensitivity']:.2f}, spec {r['specificity']:.2f}" for _, r in o.iterrows()]

    methods = (
        f"**Cross-validation.** Performance was estimated with {k}-fold cross-validation at the patient level "
        f"(StratifiedGroupKFold, shuffled): all slides of a patient were in the same fold and folds were stratified "
        f"by label. Within each training split, a patient-level validation subset was used only for early stopping. "
        f"Every slide was scored once, by the model that had not seen it (out-of-fold predictions). "
        f"{'Each slide was represented by up to ' + format(int(cfg['max_tiles']), ',') + ' randomly sampled tiles. ' if cfg.get('max_tiles') else ''}\n\n"
        f"**Statistics.** A fixed threshold of 0.5 on P({pos_name}) was used for all organs. Out-of-fold predictions "
        f"were pooled to compute AUC, balanced accuracy, sensitivity, specificity, PPV and NPV, overall and per organ. "
        f"95% CIs are percentile bootstrap intervals ({n_boot} resamples) resampling patients, not slides. We report "
        f"the mean ± SD of AUC over test folds and a within-organ AUC (slide-weighted mean of per-organ AUCs), which "
        f"removes differences between organs from the pooled estimate. Heterogeneity across organs was assessed with "
        f"Cochran's Q and I² using bootstrap standard errors."
        + (" As a baseline, slides were ranked by tile count alone (amount of tissue)." if has_base else "")
        + (f" {n_skip} unreadable slides were excluded." if n_skip else ""))

    results = (
        f"Across {k}-fold cross-validation on {int(allr['n_slides'])} slides from {int(allr['n_patients'])} patients and "
        f"{len(o)} organs, the classifier reached an AUC of {fa.mean():.3f} ± {fa.std(ddof=0):.3f} (mean ± SD over folds; "
        f"Fig. 1). Pooled over all out-of-fold predictions, AUC was {_ci(allr, 'auc')}, with sensitivity "
        f"{allr['sensitivity']:.3f} and specificity {allr['specificity']:.3f} at a threshold of 0.5. Per organ, AUC ranged "
        f"from {worst['auc']:.3f} ({worst[group_col]}) to {best['auc']:.3f} ({best[group_col]}) (Table 2, Fig. 2). The "
        f"within-organ AUC was {within:.3f}, "
        + ("close to the pooled AUC, so the pooled estimate is not driven by differences between organs."
           if abs(diff) < 0.01 else
           "below the pooled AUC, so part of the pooled estimate reflects differences between organs."
           if diff > 0 else "above the pooled AUC.")
        + (f" Tile count alone reached an AUC of {allr['base_auc']:.3f}." if has_base else ""))

    legends = (
        f"**Table 1 | Cohort.** Slides and patients per organ, {pos_name} and {neg_name} slides, slides per patient.\n\n"
        f"**Table 2 | Performance per organ ({k}-fold cross-validation).** Pooled out-of-fold predictions; 95% CI from a "
        f"patient-level bootstrap ({n_boot} resamples); threshold 0.5 for all organs. Within-organ AUC is the "
        f"slide-weighted mean of per-organ AUCs.\n\n"
        f"**Fig. 1 | Cross-validated ROC.** Thin lines, the {k} held-out test folds; thick line, pooled out-of-fold "
        f"predictions; dot, operating point at 0.5" + ("; dashed, tile count alone" if has_base else "") + ".\n\n"
        f"**Fig. 2 | Performance per organ.** AUC, sensitivity and specificity with 95% CI (patient-level bootstrap). "
        f"Dashed line, all organs pooled; dotted line, AUC 0.5. n, slides ({pos_name}/{neg_name}).\n\n"
        f"**Supplementary Table | Per fold.** Size and performance of each test fold, mean ± SD.\n\n"
        f"**Supplementary Table | Slides per organ and fold.** Every organ is represented in each test fold.\n\n"
        f"**Supplementary Fig. | ROC per organ.** Blue, organ; grey, all organs pooled; dot, threshold 0.5.\n\n"
        f"**Supplementary Fig. | Predicted probabilities.** Out-of-fold P({pos_name}) per organ by true class; lines, "
        f"median and IQR; dashed, threshold 0.5.\n\n"
        f"**Supplementary Fig. | Confusion matrices.** Per organ and pooled, threshold 0.5; counts and row percentages.")

    return ("# 5-fold CV across organs — numbers and draft text (check every sentence)\n\n## Key numbers\n"
            + "\n".join(numbers) + "\n\n## Methods\n" + methods + "\n\n## Results\n" + results
            + "\n\n## Legends\n" + legends + "\n")
