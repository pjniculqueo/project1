import numpy as np

def collect_data(num_samples = 1000, num_features = 1, seed =27) -> np.ndarray:
    """Generate random data to simulate market returns."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(num_samples, num_features))

if __name__ == "__main__":
    print(collect_data().shape)
