from capacity_panel import plot_capacity_panel


plot_capacity_panel(
    csv_path="data/gridSearch_SomaNoise/coeff_infty/synapse_noise_0.3/max_points_df.csv",
    output_file="fig4d.pdf",
    panel_title="Synaptic noise",
    show_ylabel=True,
)
