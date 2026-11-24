"""
One figure style for all paper figures (cv_paper_results, visualize_features, ...). Importing applies it.

    from paper_style import MM, PALETTE
    fig, ax = plt.subplots(figsize=(89 * MM, 89 * MM))
    ax.scatter(x, y, color=PALETTE[i % len(PALETTE)])
"""

import matplotlib.pyplot as plt

MM = 1 / 25.4  # Nature: 89 mm single, 183 mm double column
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
MAIN_C, POS_C, NEG_C, BASE_C = "#2a78d6", "#eb6834", "#2a78d6", "#eb6834"
EXTRA_C = ["#1a9e77", "#9b4dca", "#d4a017", "#d6336c", "#5c5c5c"]  # extra curves / categories
PALETTE = [MAIN_C, POS_C] + EXTRA_C  # categorical colors, in this order

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150,
})
