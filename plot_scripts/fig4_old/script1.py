import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import UnivariateSpline

bk_spacing = 0.25
b_spacing = 0.15
grid_search_bk_size = 5
grid_search_b_size = 13

ndRlist = [1, 2, 3, 4, 5, 10, 20, 30, 60, 100, 300]
ndlist = [1, 2, 3, 4, 5, 10, 12, 15, 20, 30, 50, 60, 75, 100, 150, 300, 400, 750, 1000, 1500]
nslist = [60000, 30000, 20000, 15000, 12000, 6000, 5000, 4000, 3000, 2000, 1200, 1000, 800, 600, 400, 200, 150, 80, 60, 40]

# ---------------------------
# User settings for this panel
# ---------------------------
OUTPUT_FILE = "fig4f.pdf"
PANEL_TITLE = r"Effect of variable $N_s$"
X_LABEL = r"$N_d$"
Y_LABEL = "Capacity"

SHOW_YLABEL = False     # set False for middle/right-column panels
SHOW_LEGEND = False    # use False if you will make one shared legend later

# ---------------------------
# Read data
# ---------------------------
# df = pd.read_csv("../../data/gridSearch_standard/60000/coeff_infty/max_points_df.csv")
# df = pd.read_csv("../../data/gridSearch_StepActivation/60000/coeff_infty/max_points_df.csv")
# df = pd.read_csv("../../data/gridSearch_sparse_binomial/60000/fA0.15_mu4_nB10/max_points_df.csv")
# df = pd.read_csv("../../data/gridSearch_SomaNoise/coeff_infty/synapse_noise_0.3/max_points_df.csv")
# df = pd.read_csv("../../data/gridSearch_SomaNoise/coeff_infty/dendrite_noise_0.35/max_points_df.csv")
df = pd.read_csv("../../data/gridSearch_variableNs/0.9/max_points_df.csv")

# ---------------------------
# Spline helper
# ---------------------------
def smooth_spline(x, y, y_std, s_factor=1, k=3):
    non_zero_idx = np.argwhere(y > 0).flatten()
    x = x[non_zero_idx]
    y = y[non_zero_idx]
    y_std = y_std[non_zero_idx]

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    y_std = np.maximum(y_std[idx], 1e-10)

    w = 1 / y_std
    sp = UnivariateSpline(x, y, w=w, k=k, s=s_factor * len(y))
    return sp

# ---------------------------
# Global style
# ---------------------------
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
})

# Optional LaTeX font matching
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
# })

# ---------------------------
# Panel size
# ---------------------------
# Designed for a final assembled 3x3 figure of about 6.8 x 6.2 inches
PANEL_SIZE = (2.1, 2.3)

fig, ax = plt.subplots(figsize=PANEL_SIZE)

# Tight but safe margins for an individually exported panel
fig.subplots_adjust(
    left=0.22 if SHOW_YLABEL else 0.10,
    right=0.98,
    bottom=0.18,
    top=0.88
)

# ---------------------------
# Plot
# ---------------------------
for ndR_idx, ndR_value in enumerate([1, 2, 3, 4, 5, 10]):
    subset = df[df["ndR"] == ndR_value].copy()
    subset = subset.loc[subset.groupby(["capacity", "nd"])["b"].idxmax()]
    subset = subset.sort_values("nd")

    if len(subset) < 4:
        print(f"Skipping ndR = {ndR_value}: too few points")
        continue

    max_nd = subset["nd"].max()
    min_nd = subset["nd"].min()

    nd_plot_list = np.linspace(min_nd, max_nd, max(int(max_nd - min_nd), 200))

    try:
        xs = np.log(subset["nd"].to_numpy())
        ys = subset["capacity"].to_numpy()
        y_sem = subset["sem"].to_numpy()

        spline = smooth_spline(xs, ys, y_sem)
        fitted_capacity = spline(np.log(nd_plot_list))

        ax.plot(
            nd_plot_list,
            fitted_capacity,
            label=f"{ndR_value}",
            color=f"C{ndR_idx}",
            linewidth=0.75,
        )
    except Exception as e:
        print(f"Spline failed for ndR = {ndR_value}: {e}")

    ax.errorbar(
        x=subset["nd"],
        y=subset["capacity"],
        yerr=np.sqrt(6) * subset["sem"],
        fmt="o",
        markersize=1.0,
        capsize=1.5,
        capthick=0.6,
        elinewidth=0.6,
        color=f"C{ndR_idx}",
    )

# ---------------------------
# Axes formatting
# ---------------------------
ax.set_xscale("log")
ax.set_title(PANEL_TITLE, pad=2)
ax.set_xlabel(X_LABEL)

if SHOW_YLABEL:
    ax.set_ylabel(Y_LABEL)
else:
    ax.set_ylabel("")

# Optional: keep y tick labels even when ylabel is removed
# If you want to remove y tick labels too for non-left panels, use:
# if not SHOW_YLABEL:
#     ax.set_yticklabels([])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", which="major", length=3)
ax.tick_params(axis="both", which="minor", length=2)

# Optional legend
if SHOW_LEGEND:
    leg = ax.legend(
        title=r"$n$",
        ncol=2,                 # 2 columns → 3 rows for 6 items
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
        borderpad=0.2,
        labelspacing=0.3,
    )

    leg.get_title().set_fontsize(8)

# ---------------------------
# Save
# ---------------------------
plt.savefig(OUTPUT_FILE, bbox_inches="tight")
plt.close()