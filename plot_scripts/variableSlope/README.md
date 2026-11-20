Run `python plot_scripts/variableSlope/capa_vs_nd.py` from the repository root
(or invoke the script by absolute path from any directory).
The PDF is saved beside the script.

Green curves compare all-one slopes, uniform slopes in [0.5, 1.5], and uniform
slopes in [0.333, 3], with solid, dashed, and dotted lines respectively.
The summaries are read from `data/`; the all-one baseline is the standard
60,000-synapse independent-input n-WTA experiment.

Choose the number of panels by listing their n values, then set the layout:

```bash
python plot_scripts/variableSlope/capa_vs_nd.py --n-values 1 2 3 4 5 10 --nrows 2 --ncols 3
```

You can also edit `PANEL_N_VALUES`, `N_ROWS`, and `N_COLS` in the script.
Unspecified dimensions are inferred (at most four columns when both are omitted).
Extra grid cells are left empty; undersized grids and unavailable n values are errors.
Use `--output PATH` for another filename or format, including PNG.
Axis limits are shared with the reference: Nd from 0.5 to 2000 on a log scale,
and capacity from 0 to 1800. Error bars show trial standard deviation
(`sqrt(n_samples) * sem`, assuming six trials for legacy summaries).
