# Tests

Run the automated Python tests from the repository root:

```bash
JAX_PLATFORMS=cpu python -B -m unittest discover -s tests -v
```

Run an individual suite with `python -B -m unittest tests.test_variable_slope -v`, `python -B -m unittest tests.test_plan_grid_search -v`, or `python -B -m unittest tests.test_initialize_grid_search -v`. The driver tests require JAX, Flax, and NumPy; the grid-search tests require NumPy. Tests use small synthetic simulations and temporary files rather than production experiment data.

The root `test*.ipynb` files are historical exploratory notebooks, not automated tests; they remain in place to preserve their existing paths and workflow.
