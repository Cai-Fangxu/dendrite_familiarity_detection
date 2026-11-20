"""Compare n-WTA readout slope ranges at fixed n (one panel per n value).

Example: python plot_scripts/variableSlope/capa_vs_nd.py --n-values 1 2 3 4 5 10 --nrows 2 --ncols 3
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from plot_scripts.capacity_panels import load_summary, make_figure, make_parser, save_figure


OUTPUT_FILE = Path(__file__).with_suffix(".pdf")
PANEL_N_VALUES = [1, 2, 3, 4, 5, 10]
N_ROWS = 2  # Infer dimensions when None; unused grid cells stay empty.
N_COLS = 3
COLOR = "C2"  # n-WTA green, as in the reference figure.
SLOPE_SPECS = [
    ("gridSearch_standard/60000/coeff_infty", "-", r"$s=1$"),
    ("gridSearch_nWTA_variableSlope/slope_0.5_1.5", "--", r"$s\sim U[0.5,1.5]$"),
    ("gridSearch_nWTA_variableSlope/slope_0.333_3", ":", r"$s\sim U[0.333,3]$"),
]


def load_curves():
    """Load the two variable-slope summaries and the all-one baseline."""
    curves = []
    for directory, linestyle, label in SLOPE_SPECS:
        df = load_summary(ROOT_DIR / "data" / directory / "max_points_df.csv", "ndR")
        curves.append((label, COLOR, linestyle, df.rename(columns={"ndR": "panel_n"})))
    return curves


def main(argv=None):
    """Render an arbitrary grid of n-WTA slope-comparison panels."""
    parser = make_parser(__doc__, PANEL_N_VALUES, N_ROWS, N_COLS, OUTPUT_FILE)
    args = parser.parse_args(argv)
    try:
        fig = make_figure(load_curves(), args.n_values, args.nrows, args.ncols)
        save_figure(fig, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
