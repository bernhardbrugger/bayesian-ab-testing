# Bayesian A/B Testing

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A lightweight Python toolkit for Bayesian A/B testing — visualize posteriors, compute credible intervals, and make decisions with probability, not p-values.

## Why Bayesian?

| Frequentist | Bayesian |
|---|---|
| p-values are confusing and often misinterpreted | Get direct probability: "B is 95% likely to be better than A" |
| Classic tests need a fixed sample size and punish peeking | The posterior is valid at any time; for stopping rules look at expected loss, not just P(best) |
| Binary reject/fail-to-reject | Rich posterior distributions show full uncertainty |
| No intuitive effect size communication | Credible intervals are what people *think* confidence intervals are |

## Quick Start

```python
from bayesian_ab import BayesianABTest

# Create test (seed makes the Monte Carlo results reproducible)
test = BayesianABTest(seed=42)

# Add variants with observed data
test.add_variant("Control", successes=120, trials=1000)
test.add_variant("Variant B", successes=145, trials=1000)

# Who's winning?
print(test.probability_of_winning())
# {'Control': 0.0491, 'Variant B': 0.9509}

# Credible intervals
print(test.credible_interval("Variant B"))
# (0.124556, 0.1682)

# Full summary
test.summary()

# Relative uplift of B over Control, with uncertainty
print(test.lift("Variant B", "Control"))
# {'mean': 0.215, 'median': 0.207, 'ci': (-0.036, 0.513), 'prob_positive': 0.95}

# Visualize posteriors
test.posterior_plot()
```

## Features

- 🎯 **Probability of winning** — Monte Carlo simulation for variant comparison
- 📊 **Posterior visualization** — Plot Beta distributions for each variant
- 📏 **Credible intervals** — equal-tailed (default) or highest density (`method="hdi"`)
- 📉 **Expected loss** — what you give up, on average, if you pick the wrong variant
- 🔄 **Incremental updates** — Add observations over time
- 📋 **Summary reports** — Human-readable test summaries
- 🧮 **Lift estimation** — relative uplift over a baseline with a credible interval
- 🎲 **Reproducible** — pass `seed=` for identical results on every run

## Installation

```bash
pip install "git+https://github.com/bernhardbrugger/bayesian-ab-testing.git"
pip install matplotlib   # only needed for posterior_plot
```

Or from a clone:

```bash
git clone https://github.com/bernhardbrugger/bayesian-ab-testing.git
cd bayesian-ab-testing
pip install -e ".[dev]"   # includes matplotlib and pytest
```

## Usage

### Basic Conversion Test

```python
from bayesian_ab import BayesianABTest

test = BayesianABTest()
test.add_variant("Homepage A", successes=510, trials=5000)
test.add_variant("Homepage B", successes=550, trials=5000)

# Probability each variant is the best
probs = test.probability_of_winning()
print(f"P(B is better) = {probs['Homepage B']:.1%}")

# Per-variant 95% credible intervals, P(best), expected loss and lift vs. the first variant
test.summary()
```

### Posterior Visualization

```python
ax = test.posterior_plot(figsize=(10, 6), show=False)   # returns the matplotlib Axes
ax.figure.savefig("posteriors.png", dpi=150)
# Overlaid Beta distribution curves for each variant with shaded credible intervals
```

### Decisions

```python
test.prob_better("Homepage B", "Homepage A")        # P(B > A)
test.expected_loss()                                # average rate you lose by choosing each variant
test.lift("Homepage B", "Homepage A")               # {"mean", "median", "ci", "prob_positive"}
test.credible_interval("Homepage B", method="hdi")  # shortest 95 % interval
```

### Incremental Updates

```python
test = BayesianABTest(prior_alpha=1, prior_beta=1)  # Uniform prior
test.add_variant("A", successes=50, trials=500)
test.add_variant("B", successes=60, trials=500)

# Day 2: more data comes in
test.add_observations("A", successes=45, trials=500)
test.add_observations("B", successes=58, trials=500)

test.summary()
```

## Mathematical Background

### Beta-Binomial Model

For conversion rate testing, we model the conversion rate θ of each variant as a Beta distribution:

**Prior:** θ ~ Beta(α₀, β₀)

With a uniform prior, α₀ = β₀ = 1 (all rates equally likely).

**Likelihood:** Given s successes in n trials: X ~ Binomial(n, θ)

**Posterior:** θ | data ~ Beta(α₀ + s, β₀ + n - s)

The conjugacy of Beta-Binomial gives us an analytical posterior — no MCMC needed.

### Probability of Winning

To compute P(B > A), we draw N samples from each posterior and count:

P(B > A) ≈ (1/N) Σ 𝟙[θ_B⁽ⁱ⁾ > θ_A⁽ⁱ⁾]

With N = 100,000 samples, this gives precise estimates.

### Credible Intervals

By default the **Equal-Tailed Interval (ETI)**: the Beta posterior's PPF (percent point function) at α/2 and 1 - α/2. With `method="hdi"` you get the **Highest Density Interval**, the shortest interval with the same probability mass. For symmetric posteriors both agree; for skewed ones (small counts, rates near 0) the HDI is shorter. It is not defined for U-shaped posteriors (both Beta parameters ≤ 1).

### Lift

Relative uplift is (θ_variant − θ_baseline) / θ_baseline, computed per Monte Carlo draw from both posteriors, so its credible interval reflects the uncertainty of both rates.

### Limits

Only binary conversion metrics (Beta-Binomial). No multi-armed-bandit allocation, no revenue/continuous metrics. P(best) and the intervals are Monte Carlo estimates (error ≈ 1/√N), so set `seed=` when you need stable numbers in a report.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests compare the Monte Carlo results with exact numerical integration of the Beta posteriors. A GitHub Actions workflow is prepared in `ci/github-actions-ci.yml`.

## Contributing

Contributions welcome! Please open an issue or submit a PR.

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE) for details.
