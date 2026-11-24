import numpy as np

from flight_delay_bioopt.data import load_config
from flight_delay_bioopt.multiseed import ORIGINAL_RESULT_SEEDS, parse_algorithm_selection
from flight_delay_bioopt.optimization import (
    ALGORITHMS,
    _run_scipy_nelder_mead,
    decode_vector,
    grid_vectors,
)


def test_decode_vector_respects_search_boundaries() -> None:
    search_space = load_config()["optimization"]["search_space"]
    low = decode_vector(np.zeros(5), search_space)
    high = decode_vector(np.ones(5), search_space)

    assert low["max_depth"] == 3
    assert high["max_depth"] == 10
    assert low["learning_rate"] == 0.02
    assert np.isclose(high["learning_rate"], 0.3)
    assert low["scale_pos_weight"] == 1.0
    assert high["scale_pos_weight"] == 6.5


def test_full_coarse_grid_has_32_unique_candidates() -> None:
    candidates = grid_vectors()
    assert len(candidates) == 32
    assert len({tuple(candidate) for candidate in candidates}) == 32


def test_scipy_nelder_mead_respects_bounds_and_budget() -> None:
    class QuadraticObjective:
        budget = 12
        calls = 0
        best_score = -1.0

        def __call__(self, vector: np.ndarray) -> float:
            if self.calls >= self.budget:
                return self.best_score
            values = np.asarray(vector)
            assert np.all(values >= 0.0)
            assert np.all(values <= 1.0)
            self.calls += 1
            score = float(1.0 - np.square(values - 0.7).sum())
            self.best_score = max(self.best_score, score)
            return score

    objective = QuadraticObjective()
    _run_scipy_nelder_mead(objective, budget=objective.budget, seed=42)

    assert objective.calls == objective.budget
    assert "scipy_nelder_mead" in ALGORITHMS


def test_multiseed_selection_and_original_seed_metadata() -> None:
    assert parse_algorithm_selection("all") == list(ALGORITHMS)
    assert parse_algorithm_selection("bayesian_tpe,genetic_algorithm") == [
        "genetic_algorithm",
        "bayesian_tpe",
    ]
    assert set(ORIGINAL_RESULT_SEEDS) == set(ALGORITHMS)
