import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import UnivariateSpline

ndRlist = [1, 2, 3, 4, 5, 10, 20, 30, 60, 100, 300]
ndlist = [1, 2, 3, 4, 5, 10, 12, 15, 20, 30, 50, 60, 75, 100, 150, 300, 400, 750, 1000, 1500]
nslist = [60000, 30000, 20000, 15000, 12000, 6000, 5000, 4000, 3000, 2000, 1200, 1000, 800, 600, 400, 200, 150, 80, 60, 40]
noise_str_list = ["0", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.6", "0.7", "0.9"]
noise_list = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.9]
noise_mode = "dendrite"
n_trials = 6
coeff = "infty"

# ---------------------------
# User settings for this panel
# ---------------------------
OUTPUT_FILE = "fig4h.pdf"
PANEL_TITLE = r"Effect of dendritic noise"
X_LABEL = r"Dendritic noise level $\sigma$"
Y_LABEL = "Capacity"

SHOW_YLABEL = False     # set False for middle/right-column panels
SHOW_LEGEND = False    # use False if you will make one shared legend later

# ---------------------------
# Read data
# ---------------------------
file_path0 = f"../../data/gridSearch_SomaNoise/coeff_{coeff}"

df_noise = pd.DataFrame(data=None)
dfs = []
# try to read the zero noise file first
try:
    tmp_df = pd.read_csv(f"../../data/gridSearch_standard/60000/coeff_{coeff}/max_points_df.csv")
    tmp_df['noise'] = "0"
    dfs.append(tmp_df)
except:
    dfs.append(pd.DataFrame())  # empty DataFrame on failure
# read other noise files
for noise_idx, noise in enumerate(noise_list[1:]):
    noise_str = noise_str_list[1:][noise_idx]
    file_path = f"{file_path0}/{noise_mode}_noise_{noise_str}/max_points_df.csv"
    try:
        tmp_df = pd.read_csv(file_path)
        tmp_df['noise'] = noise_str
        dfs.append(tmp_df)
    except:
        dfs.append(pd.DataFrame())  # empty DataFrame on failure

df = pd.concat(dfs, ignore_index=True)


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
max_df = df.loc[df.groupby(['noise', 'ndR'])['capacity'].idxmax()]
for ndR_idx, ndR in enumerate(ndRlist):
    subset_ndR = max_df[max_df['ndR'] == ndR]
    subset_ndR = subset_ndR.sort_values("noise")
    plt.plot(subset_ndR['noise'].to_numpy().astype(float), subset_ndR['capacity'], label=f"{ndR}", c=f"C{ndR_idx}", linewidth=0.75)
    plt.errorbar(
        x=subset_ndR['noise'].to_numpy().astype(float),
        y=subset_ndR['capacity'],
        yerr=np.sqrt(n_trials) * subset_ndR['sem'], 
        fmt='o',  
        markersize=1.0,
        capsize=1.5,
        capthick=0.6,
        elinewidth=0.6, 
        c=f"C{ndR_idx}"
    )

# ---------------------------
# Axes formatting
# ---------------------------
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