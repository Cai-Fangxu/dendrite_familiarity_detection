from capacity_panel import plot_capacity_panel


plot_capacity_panel(
    csv_path="data/gridSearch_standard/60000/coeff_infty/max_points_df.csv",
    output_file="fig4a.pdf",
    panel_title="Original model",
    show_ylabel=True,
    show_legend=True,
)
