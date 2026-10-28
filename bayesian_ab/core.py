"""Core Bayesian A/B testing implementation (Beta-Binomial model)."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


def _check_count(value, label: str) -> int:
    if isinstance(value, bool) or not float(value).is_integer() or value < 0:
        raise ValueError(f"{label} must be a non-negative whole number, got {value!r}")
    return int(value)


class BayesianABTest:
    """Bayesian A/B test using a Beta-Binomial conjugate model.

    Parameters
    ----------
    prior_alpha : float
        Alpha parameter of the Beta prior (default: 1 = uniform).
    prior_beta : float
        Beta parameter of the Beta prior (default: 1 = uniform).
    n_samples : int
        Number of Monte Carlo samples for probability estimation.
    seed : int, optional
        Seed for the random number generator. With a seed, repeated runs of the same
        script give identical results.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 n_samples: int = 100_000, seed: Optional[int] = None):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_samples = int(n_samples)
        self.variants: Dict[str, Dict[str, float]] = {}
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ data

    def add_variant(self, name: str, successes: int, trials: int) -> None:
        """Add a variant with observed data.

        Raises ValueError for negative or non-integer counts, ``trials < successes`` and
        for a name that already exists (use :meth:`add_observations` to add more data).
        """
        successes = _check_count(successes, "successes")
        trials = _check_count(trials, "trials")
        if trials < successes:
            raise ValueError("Trials must be >= successes")
        if name in self.variants:
            raise ValueError(f"Variant '{name}' already exists. Use add_observations to add data.")
        self.variants[name] = {
            "alpha": self.prior_alpha + successes,
            "beta": self.prior_beta + trials - successes,
            "successes": successes,
            "trials": trials,
        }

    def add_observations(self, name: str, successes: int, trials: int) -> None:
        """Add more observations (``successes`` out of ``trials``) to an existing variant."""
        if name not in self.variants:
            raise KeyError(f"Variant '{name}' not found. Use add_variant first.")
        successes = _check_count(successes, "successes")
        trials = _check_count(trials, "trials")
        if trials < successes:
            raise ValueError("Trials must be >= successes")
        v = self.variants[name]
        v["alpha"] += successes
        v["beta"] += trials - successes
        v["successes"] += successes
        v["trials"] += trials

    # ------------------------------------------------------------- internals

    def _get_posterior(self, name: str):
        """The Beta posterior distribution of a variant."""
        if name not in self.variants:
            raise KeyError(f"Variant '{name}' not found.")
        v = self.variants[name]
        return stats.beta(v["alpha"], v["beta"])

    def _sample_posteriors(self) -> Dict[str, np.ndarray]:
        """Draw Monte Carlo samples from all variant posteriors."""
        return {
            name: self._rng.beta(v["alpha"], v["beta"], size=self.n_samples)
            for name, v in self.variants.items()
        }

    def _pair_samples(self, a: str, b: str) -> Tuple[np.ndarray, np.ndarray]:
        for n in (a, b):
            if n not in self.variants:
                raise KeyError(f"Variant '{n}' not found.")
        va, vb = self.variants[a], self.variants[b]
        return (self._rng.beta(va["alpha"], va["beta"], size=self.n_samples),
                self._rng.beta(vb["alpha"], vb["beta"], size=self.n_samples))

    # ------------------------------------------------------------- decisions

    def probability_of_winning(self) -> Dict[str, float]:
        """Probability that each variant has the highest true conversion rate."""
        if len(self.variants) < 2:
            raise ValueError("Need at least 2 variants")
        samples = self._sample_posteriors()
        names = list(samples.keys())
        stacked = np.column_stack([samples[n] for n in names])
        winners = np.argmax(stacked, axis=1)
        counts = np.bincount(winners, minlength=len(names))
        return {name: float(counts[i] / self.n_samples) for i, name in enumerate(names)}

    def prob_better(self, a: str, b: str) -> float:
        """P(rate of ``a`` > rate of ``b``) for two named variants."""
        sa, sb = self._pair_samples(a, b)
        return float(np.mean(sa > sb))

    def expected_loss(self) -> Dict[str, float]:
        """Expected loss (in absolute conversion rate) of choosing each variant.

        The loss of variant V is the average amount by which the best variant beats V
        under the posterior; the variant with the smallest loss is the safest choice.
        """
        samples = self._sample_posteriors()
        names = list(samples.keys())
        stacked = np.column_stack([samples[n] for n in names])
        best = np.max(stacked, axis=1)
        return {name: float(np.mean(best - samples[name])) for name in names}

    def lift(self, variant: str, baseline: str, alpha: float = 0.05) -> Dict[str, object]:
        """Relative uplift of ``variant`` over ``baseline``: (rate_variant - rate_base) / rate_base.

        Returns a dict with the posterior ``mean`` and ``median`` lift, an equal-tailed
        credible interval ``ci`` at level ``1 - alpha`` and ``prob_positive`` (the probability
        that the variant is better). Computed by Monte Carlo from the two posteriors.
        """
        sv, sb = self._pair_samples(variant, baseline)
        rel = (sv - sb) / sb
        lo, hi = np.quantile(rel, [alpha / 2, 1 - alpha / 2])
        return {
            "mean": float(np.mean(rel)),
            "median": float(np.median(rel)),
            "ci": (float(lo), float(hi)),
            "prob_positive": float(np.mean(sv > sb)),
        }

    # ------------------------------------------------------------- intervals

    def credible_interval(self, name: str, alpha: float = 0.05,
                          method: str = "eti") -> Tuple[float, float]:
        """Credible interval of a variant's conversion rate at level ``1 - alpha``.

        method="eti" (default): equal-tailed interval, the quantiles at alpha/2 and 1 - alpha/2.
        method="hdi": highest density interval, the shortest interval holding ``1 - alpha``
        of the posterior mass. Not defined for U-shaped posteriors (both Beta parameters <= 1),
        where it raises ValueError.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        posterior = self._get_posterior(name)
        if method == "eti":
            lower, upper = posterior.ppf(alpha / 2), posterior.ppf(1 - alpha / 2)
        elif method == "hdi":
            lower, upper = self._hdi(posterior, alpha)
        else:
            raise ValueError("method must be 'eti' or 'hdi'")
        return (round(float(lower), 6), round(float(upper), 6))

    @staticmethod
    def _hdi(dist, alpha: float) -> Tuple[float, float]:
        a, b = dist.args
        if a <= 1 and b <= 1:
            raise ValueError("The posterior is U-shaped; a highest density interval is not "
                             "a single interval. Use method='eti'.")
        from scipy.optimize import minimize_scalar

        def width(p):
            return dist.ppf(p + 1 - alpha) - dist.ppf(p)

        res = minimize_scalar(width, bounds=(0.0, alpha), method="bounded",
                              options={"xatol": 1e-12})
        # the optimum can sit exactly on the boundary (J-shaped posteriors)
        p = min((res.x, 0.0, alpha), key=width)
        return dist.ppf(p), dist.ppf(p + 1 - alpha)

    # ---------------------------------------------------------------- output

    def posterior_plot(self, figsize: Tuple[int, int] = (10, 6),
                       credible_level: float = 0.95, ax=None, show: bool = True):
        """Plot the posterior distributions of all variants and return the matplotlib Axes.

        Parameters
        ----------
        figsize : tuple
            Figure size (ignored when ``ax`` is given).
        credible_level : float
            Credible interval level to shade.
        ax : matplotlib Axes, optional
            Draw into an existing Axes.
        show : bool
            Call ``plt.show()`` (default). Pass False to save or embed the figure yourself.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("posterior_plot needs matplotlib: pip install matplotlib "
                              "(or pip install bayesian-ab-testing[plot])") from e

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        alpha = 1 - credible_level

        for name, v in self.variants.items():
            dist = stats.beta(v["alpha"], v["beta"])
            x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 1000)
            y = dist.pdf(x)
            line, = ax.plot(x, y, label=name, linewidth=2)

            # Shade credible interval
            ci_low, ci_high = dist.ppf(alpha / 2), dist.ppf(1 - alpha / 2)
            mask = (x >= ci_low) & (x <= ci_high)
            ax.fill_between(x[mask], y[mask], alpha=0.2, color=line.get_color())

        ax.set_xlabel("Conversion Rate", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Posterior Distributions", fontsize=14)
        ax.legend(fontsize=11)
        ax.figure.tight_layout()
        if show:
            plt.show()
        return ax

    def summary(self) -> str:
        """Print a human-readable summary of the test and return it as text.

        The first variant added is treated as the baseline for the lift lines.
        """
        probs = self.probability_of_winning()
        losses = self.expected_loss()
        lines = ["=" * 60, "Bayesian A/B Test Summary", "=" * 60, ""]
        baseline = next(iter(self.variants))

        for name, v in self.variants.items():
            ci = self.credible_interval(name)
            rate = f"{v['successes'] / v['trials']:.4f}" if v["trials"] else "n/a"
            lines.append(f"  {name}:")
            lines.append(f"    Conversion rate: {rate} ({v['successes']}/{v['trials']})")
            lines.append(f"    95% Credible Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
            lines.append(f"    P(best): {probs[name]:.4f}")
            lines.append(f"    Expected loss: {losses[name]:.6f}")
            if name != baseline:
                lift = self.lift(name, baseline)
                lines.append(f"    Lift vs {baseline}: {lift['mean']:+.1%} "
                             f"(95% CI {lift['ci'][0]:+.1%} to {lift['ci'][1]:+.1%})")
            lines.append("")

        winner = max(probs, key=probs.get)
        lines.append(f"  → Winner: {winner} (P={probs[winner]:.1%})")
        lines.append("=" * 60)

        text = "\n".join(lines)
        print(text)
        return text
