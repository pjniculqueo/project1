import numpy as np


def calibrate_parameters(data: np.ndarray) -> dict:
    """Calibrate mean vector and covariance/correlation matrices."""
    mu = data.mean(axis=0)
    cov = np.cov(data, rowvar=False)
    corr = np.corrcoef(data, rowvar=False)
    return {"mu": mu, "cov": cov, "corr": corr}

if __name__ == "__main__":
    import sys
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    data = np.random.default_rng(0).normal(size=(size, 1))
    params = calibrate_parameters(data)
    print(params)
