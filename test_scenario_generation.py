import numpy as np
from scenario_generation import build_scenarios


def test_build_scenarios_shape():
    params = {"mu": np.zeros(2), "cov": np.eye(2)}
    scenarios = build_scenarios(params, num_scenarios=5, seed=1)
    assert scenarios.shape == (5, 2)
