import numpy as np


def build_scenarios(params: dict, num_scenarios: int = 100, seed: int | None = None) -> np.ndarray:
    """Generate scenarios from calibrated parameters."""
    mu = params["mu"]
    cov = params["cov"]
    if np.ndim(cov) == 0:
        cov = np.array([[cov]])
        mu = np.asarray(mu).reshape(1)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=num_scenarios)

if __name__ == "__main__":
    from data_collection import collect_data
    from parameter_calibration import calibrate_parameters

    data = collect_data()
    params = calibrate_parameters(data)
    scenarios = build_scenarios(params)
    print(scenarios.shape)
