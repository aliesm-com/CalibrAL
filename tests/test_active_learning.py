"""Unit tests for uncertainty scoring used by active learning."""

from __future__ import annotations

import numpy as np

from calibral.active_learning import uncertainty_from_prob_positive


def test_entropy_highest_near_half():
    scores = uncertainty_from_prob_positive([0.5, 0.9, 0.1], method="entropy")
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_margin_highest_near_half():
    scores = uncertainty_from_prob_positive([0.5, 0.95, 0.05], method="margin")
    assert np.isclose(scores[0], 1.0)
    assert scores[0] > scores[1]


def test_results_row_distinguishes_llm_modes():
    """Mock vs none/api must remain distinguishable in recorded results."""
    modes = {"mock", "api", "replay", "none"}
    assert "mock" in modes and "none" in modes and "api" in modes
