# Changelog

## 0.2.0
- **New:** `lift()` (relative uplift with credible interval), `prob_better(a, b)`, `credible_interval(..., method="hdi")` (the README promised an HDI, the code only had equal-tailed intervals), `seed=` for reproducible results.
- **New:** `posterior_plot()` returns the Axes and accepts `ax=` and `show=False`, so figures can be saved or embedded.
- **New:** `summary()` shows the lift against the first variant and no longer crashes for a variant with zero trials.
- **Validation:** negative or fractional counts, invalid priors and failed `add_observations` calls (which used to change the data before failing) are rejected. Adding a variant name twice now raises instead of silently overwriting the data. *This can break code that relied on overwriting.*
- matplotlib is optional (`pip install bayesian-ab-testing[plot]`); packaging moved to `pyproject.toml`.
- 33 tests, including checks against exact numerical integration of the Beta posteriors.

## 0.1.0
- First version.
