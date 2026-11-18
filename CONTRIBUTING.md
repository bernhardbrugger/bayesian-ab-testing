# Contributing to bayesian-ab-testing

Bug reports, wrong numbers and pull requests are all welcome.

## Set up and run the tests

```bash
git clone https://github.com/bernhardbrugger/bayesian-ab-testing.git
cd bayesian-ab-testing
python -m venv .venv            # .venv/Scripts/activate on Windows, source .venv/bin/activate elsewhere
pip install -e ".[dev]"         # numpy, scipy, matplotlib and pytest
pytest -q                       # or a single file: pytest -q tests/test_core.py
```

CI runs the same `pytest` on Linux and Windows with Python 3.9 and 3.12, so do not use syntax newer than 3.9.

## Reporting a bug

Open an issue with:

- a short script that shows the problem: the successes and trials per variant, the `seed=` and `n_samples=` you used, and the call,
- the output you got and the output you expected (and where the expected number comes from, for example a closed-form result or another tool),
- the package version (`python -c "import bayesian_ab; print(bayesian_ab.__version__)"`), your Python version and OS.

P(best), the intervals and the lift are Monte Carlo estimates. Two runs without a seed differ by roughly 1/√`n_samples`; that alone is not a bug, a difference larger than that is.

## Pull requests

- One change per pull request, with a test that fails without it.
- New statistics are checked against an independent result, not against the code's own output: the existing tests compare the Monte Carlo numbers with exact numerical integration of the Beta posteriors (`scipy`), and `decision()` is checked against a simulation. Use a fixed `seed=` and a tolerance that matches the Monte Carlo error.
- Invalid input raises a clear error before any data is changed (see `tests/test_extended.py`).
- Add a line under `## Unreleased` in `CHANGELOG.md` when users would notice the change, and update the README if an example's output changes (run the example and paste the real output).
- The core depends only on numpy and scipy; matplotlib stays optional (`[plot]`).
- Commit messages say what changed and why ("fix: ...", "docs: ...").

## Out of scope

The Limits section of the README lists what the package does not do (only binary conversion metrics, no bandit allocation, no revenue or continuous metrics). Please open an issue first if you want to change that, so we can agree on the approach.

By contributing you agree that your work is released under the MIT licence of this project.
