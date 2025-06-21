import numpy as np


def estimate_lambda(scenarios: np.ndarray, num_iterations: int = 1000, learning_rate: float = 0.01) -> np.ndarray:
    """Estimate a shared lambda parameter using a simple adversarial approach."""
    rng = np.random.default_rng()
    generator_param = rng.normal()
    discriminator_weight = rng.normal()

    for _ in range(num_iterations):
        idx = rng.integers(0, len(scenarios))
        real = scenarios[idx].mean()
        fake = generator_param

        d_real = 1 / (1 + np.exp(-discriminator_weight * real))
        d_fake = 1 / (1 + np.exp(-discriminator_weight * fake))
        discriminator_weight += learning_rate * (real * (1 - d_real) - fake * d_fake)

        d_fake = 1 / (1 + np.exp(-discriminator_weight * fake))
        generator_param += learning_rate * discriminator_weight * (1 - d_fake)

    lambda_t = np.full(len(scenarios), generator_param)
    return lambda_t

if __name__ == "__main__":
    from scenario_generation import build_scenarios
    params = {"mu": np.zeros(1), "cov": np.eye(1)}
    scenarios = build_scenarios(params, 10)
    print(estimate_lambda(scenarios))
