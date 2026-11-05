"""Validation, reproducibility, numerical accuracy, HDI, lift and plotting."""

import numpy as np
import pytest
from scipy import integrate, stats

from bayesian_ab import BayesianABTest


def make(seed=1, n=200_000, **kw):
    t = BayesianABTest(seed=seed, n_samples=n, **kw)
    t.add_variant("A", successes=100, trials=1000)
    t.add_variant("B", successes=130, trials=1000)
    return t


# ── validation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("successes, trials", [(-1, 10), (1, -10), (1.5, 10), (1, 10.5), (True, 10)])
def test_bad_counts_are_rejected(successes, trials):
    with pytest.raises(ValueError):
        BayesianABTest().add_variant("A", successes, trials)


def test_duplicate_variant_name_is_rejected():
    t = BayesianABTest()
    t.add_variant("A", 1, 10)
    with pytest.raises(ValueError, match="add_observations"):
        t.add_variant("A", 2, 10)


def test_add_observations_is_validated():
    t = BayesianABTest()
    t.add_variant("A", 1, 10)
    with pytest.raises(ValueError):
        t.add_observations("A", 5, 3)
    assert t.variants["A"]["trials"] == 10          # unchanged after the failed call


@pytest.mark.parametrize("kw", [{"prior_alpha": 0}, {"prior_beta": -1}, {"n_samples": 0}])
def test_bad_constructor_arguments(kw):
    with pytest.raises(ValueError):
        BayesianABTest(**kw)


def test_unknown_variant_names_raise_keyerror():
    t = make()
    for call in (lambda: t.credible_interval("X"), lambda: t.prob_better("A", "X"),
                 lambda: t.lift("X", "A")):
        with pytest.raises(KeyError):
            call()


# ── reproducibility ────────────────────────────────────────────────────────

def test_same_seed_gives_identical_results():
    assert make(seed=7).probability_of_winning() == make(seed=7).probability_of_winning()
    assert make(seed=7).expected_loss() == make(seed=7).expected_loss()


def test_different_seeds_differ_slightly():
    assert make(seed=1).probability_of_winning() != make(seed=2).probability_of_winning()


# ── numerical accuracy ─────────────────────────────────────────────────────

def exact_prob_a_beats_b(a, b):
    """P(X > Y) for X ~ Beta(a), Y ~ Beta(b) by numerical integration."""
    value, _ = integrate.quad(lambda x: stats.beta.pdf(x, *a) * stats.beta.cdf(x, *b), 0, 1,
                              limit=200)
    return value


def test_probability_of_winning_matches_exact_integral():
    t = make(n=400_000)
    exact_b = exact_prob_a_beats_b((1 + 130, 1 + 870), (1 + 100, 1 + 900))
    probs = t.probability_of_winning()
    assert probs["B"] == pytest.approx(exact_b, abs=0.004)
    assert probs["A"] == pytest.approx(1 - exact_b, abs=0.004)
    assert t.prob_better("B", "A") == pytest.approx(exact_b, abs=0.004)


def test_expected_loss_of_the_best_variant_is_small_and_losses_are_nonnegative():
    t = make()
    losses = t.expected_loss()
    assert min(losses.values()) >= 0
    assert losses["B"] < losses["A"]
    # with two variants, loss(A) = E[(B - A)+]: positive and of the order of the rate gap
    assert 0.02 < losses["A"] < 0.04


def test_credible_interval_matches_beta_quantiles():
    t = make()
    lo, hi = t.credible_interval("A")
    assert lo == pytest.approx(stats.beta.ppf(0.025, 101, 901), abs=1e-6)
    assert hi == pytest.approx(stats.beta.ppf(0.975, 101, 901), abs=1e-6)


# ── HDI ────────────────────────────────────────────────────────────────────

def test_hdi_equals_eti_for_symmetric_posteriors():
    t = BayesianABTest()
    t.add_variant("S", successes=49, trials=98)         # Beta(50, 50), symmetric
    assert t.credible_interval("S", method="hdi") == pytest.approx(t.credible_interval("S"), abs=1e-4)


def test_hdi_is_shorter_for_skewed_posteriors_and_holds_the_right_mass():
    t = BayesianABTest()
    t.add_variant("K", successes=3, trials=40)          # Beta(4, 38), right-skewed
    hdi, eti = t.credible_interval("K", method="hdi"), t.credible_interval("K")
    assert hdi[1] - hdi[0] < eti[1] - eti[0]
    mass = stats.beta.cdf(hdi[1], 4, 38) - stats.beta.cdf(hdi[0], 4, 38)
    assert mass == pytest.approx(0.95, abs=1e-4)


def test_hdi_for_j_shaped_posterior_starts_at_zero():
    t = BayesianABTest()
    t.add_variant("Z", successes=0, trials=30)          # Beta(1, 31): mode at 0
    lo, hi = t.credible_interval("Z", method="hdi")
    assert lo == 0 and hi == pytest.approx(stats.beta.ppf(0.95, 1, 31), abs=1e-4)


def test_hdi_rejects_u_shaped_posteriors_and_unknown_methods():
    t = BayesianABTest(prior_alpha=0.5, prior_beta=0.5)
    t.add_variant("U", successes=0, trials=0)           # Beta(0.5, 0.5)
    with pytest.raises(ValueError, match="U-shaped"):
        t.credible_interval("U", method="hdi")
    with pytest.raises(ValueError, match="method"):
        t.credible_interval("U", method="nope")
    with pytest.raises(ValueError, match="alpha"):
        t.credible_interval("U", alpha=1.5)


# ── lift ───────────────────────────────────────────────────────────────────

def test_lift_estimates_the_relative_uplift():
    t = make(n=300_000)
    lift = t.lift("B", "A")
    assert lift["mean"] == pytest.approx(0.30, abs=0.03)       # 13 % vs 10 %
    lo, hi = lift["ci"]
    assert lo < 0.30 < hi and lo < lift["median"] < hi
    assert 0.9 < lift["prob_positive"] < 1.0


def test_lift_of_a_variant_against_itself_is_centred_on_zero():
    t = make(n=300_000)
    lift = t.lift("A", "A")
    assert abs(lift["median"]) < 0.02
    assert lift["prob_positive"] == pytest.approx(0.5, abs=0.01)


# ── output ─────────────────────────────────────────────────────────────────

def test_summary_handles_zero_trials_and_reports_lift(capsys):
    t = make()
    t.add_variant("C", successes=0, trials=0)
    text = t.summary()
    assert "n/a" in text and "Lift vs A" in text and "Winner" in text
    assert capsys.readouterr().out.strip() == text.strip()


def test_posterior_plot_returns_axes_without_showing():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = make().posterior_plot(show=False)
    assert len(ax.lines) == 2 and ax.get_xlabel() == "Conversion Rate"


# ── stopping rule (expected loss) ──────────────────────────────────────────

def test_decision_waits_for_enough_data_then_stops_on_a_clear_winner():
    t = BayesianABTest(seed=3, n_samples=50_000)
    t.add_variant("A", successes=10, trials=50)
    t.add_variant("B", successes=30, trials=50)
    early = t.decision(threshold=0.01, min_trials=100)
    assert early["action"] == "continue" and "fewer than 100 trials" in early["reason"]

    t.add_observations("A", successes=100, trials=950)
    t.add_observations("B", successes=300, trials=950)
    done = t.decision(threshold=0.01, min_trials=100)
    assert done["action"] == "stop" and done["leader"] == "B"
    assert done["expected_loss"] < done["threshold"]


def test_decision_keeps_going_while_the_loss_is_above_the_threshold():
    t = BayesianABTest(seed=3, n_samples=50_000)
    t.add_variant("A", successes=100, trials=1000)
    t.add_variant("B", successes=104, trials=1000)  # nearly the same: still unsure
    res = t.decision(threshold=0.0005, min_trials=100)
    assert res["action"] == "continue" and "not yet below" in res["reason"]


def test_decision_validates_its_arguments():
    t = BayesianABTest()
    t.add_variant("A", 1, 10)
    with pytest.raises(ValueError, match="at least 2"):
        t.decision()
    t.add_variant("B", 2, 10)
    with pytest.raises(ValueError, match="threshold"):
        t.decision(threshold=0)
    with pytest.raises(ValueError):
        t.decision(min_trials=-1)


def test_the_loss_you_accept_holds_even_when_you_peek_after_every_batch():
    # Theory: if the true rates are drawn from the prior, the average regret of the variant you
    # pick when you stop is below the threshold, whatever the stopping rule. Simulated here with
    # 150 tests that look at the data after every batch of 100 visitors (they stop after about
    # five looks on average, so the peeking is real) and a prior that matches the simulation.
    rng = np.random.default_rng(2026)
    threshold, regrets, looks = 0.002, [], []
    for k in range(150):
        true = rng.beta(20, 180, 2)
        t = BayesianABTest(prior_alpha=20, prior_beta=180, seed=k, n_samples=4_000)
        t.add_variant("A", 0, 0)
        t.add_variant("B", 0, 0)
        for n_looks in range(1, 81):
            for name, p in zip("AB", true):
                t.add_observations(name, int(rng.binomial(100, p)), 100)
            res = t.decision(threshold=threshold, min_trials=100)
            if res["action"] == "stop":
                break
        looks.append(n_looks)
        regrets.append(true.max() - true[0 if res["leader"] == "A" else 1])
    assert 2 < np.mean(looks) < 15 and max(looks) < 80  # it did peek, and it did stop
    assert np.mean(regrets) < threshold + 0.001  # Monte Carlo noise allowance
