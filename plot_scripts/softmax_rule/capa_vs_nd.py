"""Plot interacting softmax capacity, with correlation encoded by linestyle.

Panels show calibrated target n_eff, not beta_select (which varies with Nd).
Example: python plot_scripts/softmax_rule/capa_vs_nd.py --n-values 1.05 2 3 4 5 10 --nrows 2 --ncols 3
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from plot_scripts.capacity_panels import load_summary, make_figure, make_parser, save_figure


OUTPUT_FILE = Path(__file__).with_suffix(".pdf")
DATA_DIR = ROOT_DIR / "data" / "gridSearch_softmax" / "int"
CALIBRATION_FILE = ROOT_DIR / "tmp_data" / "softmax_calibrations.json"
PANEL_N_VALUES = [1.05, 2, 3, 4, 5, 10]  # The calibrated near-WTA target is 1.05, not 1.
N_ROWS = 2
N_COLS = 3
# Set (lower, upper) capacity bounds by target n; omitted values use (0, 1800).
PANEL_Y_LIMITS = {
    1.05: (0, 1200),
    2: (0, 600),
    3: (0, 600),
    4: (0, 600),
    5: (0, 600),
    10: (0, 600),
}
COLOR = "C4"  # Purple distinguishes softmax from the three original rules.
COEFF_SPECS = [
    ("infty", "-", r"$c=\infty$"),
    ("4.0", "--", r"$c=4$"),
    ("1.0", ":", r"$c=1$"),
]


def load_calibrations(path):
    """Read target-n mappings from the saved calibration registry or batch report."""
    data = json.loads(Path(path).read_text())
    records = data.get("calibrations") if isinstance(data, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty 'calibrations' list")
    mapping = {}
    for record in records:
        try:
            nd, beta, target = (float(record[key])
                                for key in ("nd", "beta_select", "target_n"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid calibration record") from exc
        if (not np.isfinite([nd, beta, target]).all() or nd < 1 or nd != int(nd)
                or beta < 0 or not 1 <= target <= nd):
            raise ValueError(f"{path}: invalid nd, beta_select, or target_n")
        key = (nd, beta)
        if key in mapping and mapping[key] != target:
            raise ValueError(f"{path}: conflicting target_n for {key}")
        mapping[key] = target
    return mapping


def attach_target_n(df, mapping):
    """Match beta values with round-trip tolerance, rejecting unmapped points."""
    targets = []
    for row in df.itertuples():
        matches = {target for (nd, beta), target in mapping.items()
                   if nd == row.nd and np.isclose(beta, row.beta_select,
                                                 rtol=1e-12, atol=1e-12)}
        if len(matches) != 1:
            raise ValueError(f"Expected one calibrated target for nd={row.nd:g}, "
                             f"beta_select={row.beta_select:g}; found {sorted(matches)}. "
                             "Supply the matching --calibration-file.")
        targets.append(matches.pop())
    return df.assign(panel_n=targets)


def load_curves(calibration_file=CALIBRATION_FILE):
    """Load curated interacting-softmax summaries and their target-n metadata."""
    mapping = load_calibrations(calibration_file)
    curves = []
    for coeff, linestyle, label in COEFF_SPECS:
        df = load_summary(DATA_DIR / f"coeff_{coeff}" / "max_points_df.csv", "beta_select")
        curves.append((f"Softmax: {label}", COLOR, linestyle, attach_target_n(df, mapping)))
    return curves


def main(argv=None):
    """Render softmax panels using an explicit saved calibration mapping."""
    parser = make_parser(__doc__, PANEL_N_VALUES, N_ROWS, N_COLS, OUTPUT_FILE)
    parser.add_argument("--calibration-file", type=Path, default=CALIBRATION_FILE,
                        help="Calibration registry or batch report (default: tmp_data/softmax_calibrations.json)")
    parser.add_argument("--ylim", type=float, nargs=3, action="append",
                        metavar=("N", "MIN", "MAX"),
                        help="Override one panel's capacity range; repeat for other panels")
    args = parser.parse_args(argv)
    try:
        panel_y_limits = dict(PANEL_Y_LIMITS)
        for n, lower, upper in args.ylim or []:
            if n not in args.n_values:
                raise ValueError(f"--ylim target n={n:g} is not in --n-values")
            panel_y_limits[n] = (lower, upper)
        fig = make_figure(load_curves(args.calibration_file), args.n_values,
                          args.nrows, args.ncols, panel_symbol=r"n_{\mathrm{eff}}",
                          panel_y_limits=panel_y_limits)
        save_figure(fig, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
