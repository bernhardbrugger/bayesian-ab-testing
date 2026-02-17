"""Core Bayesian A/B testing implementation."""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple


class BayesianABTest:
    """Bayesian A/B test using Beta-Binomial conjugate model.

    Parameters
    ----------
    prior_alpha : float
        Alpha parameter of the Beta prior (default: 1 = uniform).
    prior_beta : float
        Beta parameter of the Beta prior (default: 1 = uniform).
    n_samples : int
        Number of Monte Carlo samples for probability estimation.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 n_samples: int = 100_000):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_samples = n_samples
        self.variants: Dict[str, Dict[str, float]] = {}

    def add_variant(self, name: str, successes: int, trials: int) -> None:
        """Add a variant with observed data.

        Parameters
        ----------
        name : str
            Variant name.
        successes : int
            Number of successes (conversions).
        trials : int
            Total number of trials.
        """
        if trials < successes:
            raise ValueError("Trials must be >= successes")
        self.variants[name] = {
            "alpha": self.prior_alpha + successes,
            "beta": self.prior_beta + trials - successes,
            "successes": successes,
            "trials": trials,
        }

    def add_observations(self, name: str, successes: int, trials: int) -> None:
        """Add more observations to an existing variant.

        Parameters
        ----------
        name : str
            Existing variant name.
        successes : int
            Additional successes.
        trials : int
            Additional trials.
        """
        if name not in self.variants:
            raise KeyError(f"Variant '{name}' not found. Use add_variant first.")
        v = self.variants[name]
        v["alpha"] += successes
        v["beta"] += trials - successes
        v["successes"] += successes
        v["trials"] += trials

    def _get_posterior(self, name: str) -> stats.beta:
        """Get the Beta posterior distribution for a variant."""
        v = self.variants[name]
        return stats.beta(v["alpha"], v["beta"])

    def _sample_posteriors(self) -> Dict[str, np.ndarray]:
        """Draw Monte Carlo samples from all variant posteriors."""
        rng = np.random.default_rng()
        return {
            name: rng.beta(v["alpha"], v["beta"], size=self.n_samples)
            for name, v in self.variants.items()
        }

    def probability_of_winning(self) -> Dict[str, float]:
        """Compute probability each variant is the best.

        Returns
        -------
        dict
            Mapping of variant name to probability of being the best.
        """
        if len(self.variants) < 2:
            raise ValueError("Need at least 2 variants")
        samples = self._sample_posteriors()
        names = list(samples.keys())
        stacked = np.column_stack([samples[n] for n in names])
        winners = np.argmax(stacked, axis=1)
        counts = np.bincount(winners, minlength=len(names))
        return {name: float(counts[i] / self.n_samples) for i, name in enumerate(names)}

    def credible_interval(self, name: str, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute equal-tailed credible interval for a variant's conversion rate.

        Parameters
        ----------
        name : str
            Variant name.
        alpha : float
            Significance level (default 0.05 for 95% CI).

        Returns
        -------
        tuple
            (lower, upper) bounds of the credible interval.
        """
        posterior = self._get_posterior(name)
        lower = posterior.ppf(alpha / 2)
        upper = posterior.ppf(1 - alpha / 2)
        return (round(float(lower), 6), round(float(upper), 6))

    def expected_loss(self) -> Dict[str, float]:
        """Compute expected loss for choosing each variant.

        Returns
        -------
        dict
            Expected loss (in conversion rate) for each variant.
        """
        samples = self._sample_posteriors()
        names = list(samples.keys())
        stacked = np.column_stack([samples[n] for n in names])
        best = np.max(stacked, axis=1)
        return {
            name: float(np.mean(best - samples[name]))
            for name in names
        }

    def posterior_plot(self, figsize: Tuple[int, int] = (10, 6),
                       credible_level: float = 0.95) -> None:
        """Plot posterior distributions for all variants.

        Parameters
        ----------
        figsize : tuple
            Figure size.
        credible_level : float
            Credible interval level to shade.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
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
        plt.tight_layout()
        plt.show()

    def summary(self) -> str:
        """Print a human-readable summary of the test.

        Returns
        -------
        str
            Summary text.
        """
        probs = self.probability_of_winning()
        losses = self.expected_loss()
        lines = ["=" * 60, "Bayesian A/B Test Summary", "=" * 60, ""]

        for name, v in self.variants.items():
            ci = self.credible_interval(name)
            rate = v["successes"] / v["trials"]
            lines.append(f"  {name}:")
            lines.append(f"    Conversion rate: {rate:.4f} ({v['successes']}/{v['trials']})")
            lines.append(f"    95% Credible Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
            lines.append(f"    P(best): {probs[name]:.4f}")
            lines.append(f"    Expected loss: {losses[name]:.6f}")
            lines.append("")

        winner = max(probs, key=probs.get)
        lines.append(f"  → Winner: {winner} (P={probs[winner]:.1%})")
        lines.append("=" * 60)

        text = "\n".join(lines)
        print(text)
        return text
