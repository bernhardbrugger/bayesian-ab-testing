"""Tests for BayesianABTest."""

import pytest
from bayesian_ab.core import BayesianABTest


class TestBayesianABTest:
    def test_add_variant(self):
        test = BayesianABTest()
        test.add_variant("A", successes=100, trials=1000)
        assert "A" in test.variants
        assert test.variants["A"]["alpha"] == 101
        assert test.variants["A"]["beta"] == 901

    def test_add_variant_invalid(self):
        test = BayesianABTest()
        with pytest.raises(ValueError):
            test.add_variant("A", successes=100, trials=50)

    def test_add_observations(self):
        test = BayesianABTest()
        test.add_variant("A", successes=100, trials=1000)
        test.add_observations("A", successes=50, trials=500)
        assert test.variants["A"]["successes"] == 150
        assert test.variants["A"]["trials"] == 1500

    def test_add_observations_missing(self):
        test = BayesianABTest()
        with pytest.raises(KeyError):
            test.add_observations("X", successes=10, trials=100)

    def test_probability_of_winning(self):
        test = BayesianABTest(n_samples=50_000)
        test.add_variant("A", successes=100, trials=1000)
        test.add_variant("B", successes=150, trials=1000)
        probs = test.probability_of_winning()
        assert len(probs) == 2
        assert abs(sum(probs.values()) - 1.0) < 0.01
        assert probs["B"] > probs["A"]

    def test_credible_interval(self):
        test = BayesianABTest()
        test.add_variant("A", successes=500, trials=5000)
        ci = test.credible_interval("A")
        assert ci[0] < 0.1 < ci[1]
        assert ci[0] > 0
        assert ci[1] < 1

    def test_expected_loss(self):
        test = BayesianABTest()
        test.add_variant("A", successes=100, trials=1000)
        test.add_variant("B", successes=150, trials=1000)
        losses = test.expected_loss()
        assert losses["A"] > losses["B"]

    def test_summary(self):
        test = BayesianABTest()
        test.add_variant("A", successes=100, trials=1000)
        test.add_variant("B", successes=120, trials=1000)
        result = test.summary()
        assert "Winner" in result

    def test_needs_two_variants(self):
        test = BayesianABTest()
        test.add_variant("A", successes=100, trials=1000)
        with pytest.raises(ValueError):
            test.probability_of_winning()
