import numpy as np
from adversarial_lambda import estimate_lambda


def test_estimate_lambda_constant():
    scenarios = np.zeros((5, 2))
    lambdas = estimate_lambda(scenarios, num_iterations=10, learning_rate=0.01)
    assert len(lambdas) == 5
    assert np.allclose(lambdas, lambdas[0])
