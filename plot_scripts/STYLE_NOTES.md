# Plot Script Style Notes

- Keep scripts standalone and runnable from any working directory by resolving paths from `Path(__file__).resolve()`.
- Write generated PDFs next to the script that creates them.
- Preserve the notebook's plotting logic closely; extract only the code needed to load data, build the figure, and save the panel.
- Use compact paper-panel sizing: about `(2.3, 1.9)` for small distribution plots and `(3.5, 2.0)` for wider response-age plots.
- Use small typography consistently:
  - font, ticks, labels, legends: `8`
  - no title unless the notebook/reference plot uses one
- Hide the top and right spines.
- Prefer line plots from histogram midpoints over filled histograms for distribution panels.
- Use the same color order and labels as the notebooks:
  - `C0`: `non-interacting`
  - `C1`: `variable-$u_{\text{target}}$`
  - `C2`: `n-WTA`
- Keep legends frameless when present.
- Match table placement, font size, and colored row text from the source notebook when a panel includes summary statistics.
- Save with `bbox_inches="tight"` and close figures after saving.
