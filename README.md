# Bayesian A/B Testing

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

A lightweight Python toolkit for Bayesian A/B testing — visualize posteriors, compute credible intervals, and make decisions with probability, not p-values.

## Why Bayesian?

| Frequentist | Bayesian |
|---|---|
| p-values are confusing and often misinterpreted | Get direct probability: "B is 94% likely to be better than A" |
| Requires fixed sample sizes upfront | Monitor results anytime without penalty |
| Binary reject/fail-to-reject | Rich posterior distributions show full uncertainty |
| No intuitive effect size communication | Credible intervals are what people *think* confidence intervals are |

## Quick Start

```python
from bayesian_ab import BayesianABTest

# Create test
test = BayesianABTest()

# Add variants with observed data
test.add_variant("Control", successes=120, trials=1000)
test.add_variant("Variant B", successes=145, trials=1000)

# Who's winning?
print(test.probability_of_winning())
# {'Control': 0.064, 'Variant B': 0.936}

# Credible intervals
print(test.credible_interval("Variant B"))
# (0.123, 0.168)

# Full summary
test.summary()

# Visualize posteriors
test.posterior_plot()
```

## Features

- 🎯 **Probability of winning** — Monte Carlo simulation for variant comparison
- 📊 **Posterior visualization** — Plot Beta distributions for each variant
- 📏 **Credible intervals** — HDI (Highest Density Interval) computation
- 🔄 **Incremental updates** — Add observations over time
- 📋 **Summary reports** — Human-readable test summaries
- 🧮 **Lift estimation** — Expected improvement with uncertainty

## Installation

```bash
pip install numpy scipy matplotlib
```

Clone and install locally:

```bash
git clone https://github.com/bernhardbrugger/bayesian-ab-testing.git
cd bayesian-ab-testing
pip install -e .
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

# 95% credible interval for the difference
test.summary()
```

### Posterior Visualization

```python
test.posterior_plot(figsize=(10, 6))
# Generates overlaid Beta distribution curves for each variant
# with shaded credible intervals
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

We use the **Equal-Tailed Interval (ETI)** from the Beta posterior's PPF (percent point function) at α/2 and 1 - α/2.

## Contributing

Contributions welcome! Please open an issue or submit a PR.

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE) for details.
