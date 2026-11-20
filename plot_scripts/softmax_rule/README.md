Run `python plot_scripts/softmax_rule/capa_vs_nd.py` from the repository root
(or invoke the script by absolute path from any directory).
The PDF is saved beside the script.

Purple curves show the interacting softmax rule from
`data/gridSearch_softmax/int/`, with solid, dashed, and dotted lines for
c = infinity, 4, and 1. The current summaries cover Nd = 75, 100, and 150.
Panels show calibrated target effective n. The near-WTA target is 1.05,
so the default panel values are 1.05, 2, 3, 4, 5, and 10 in a 2-by-3 grid.

The summaries contain beta_select rather than target n. The script reads the
mapping from `tmp_data/softmax_calibrations.json`; preserve that registry with
the data, or pass a matching registry/batch report using `--calibration-file PATH`.
It rejects missing or ambiguous mappings instead of estimating n from beta_select.

Choose the number of panels by listing their target n values, then set the layout:

```bash
python plot_scripts/softmax_rule/capa_vs_nd.py --n-values 1.05 2 3 4 5 10 --nrows 2 --ncols 3
```

You can also edit `PANEL_N_VALUES`, `N_ROWS`, and `N_COLS` in the script.
Unspecified dimensions are inferred (at most four columns when both are omitted).
Extra grid cells are left empty; undersized grids and unavailable n values are errors.
Use `--output PATH` for another filename or format, including PNG.
Edit `PANEL_Y_LIMITS` to set a `(lower, upper)` capacity range for each target n.
Missing entries default to `(0, 1800)`. You can also override individual panels
on the command line (repeat `--ylim N MIN MAX`):

```bash
python plot_scripts/softmax_rule/capa_vs_nd.py --ylim 1.05 800 1200 --ylim 3 200 500
```

Different ranges have independent y axes with tick labels on every panel.
Nd remains from 0.5 to 2000 on a log scale. Error bars show trial standard deviation
(`sqrt(n_samples) * sem`).
