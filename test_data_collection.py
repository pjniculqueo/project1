import numpy as np
from data_collection import collect_data

def test_collect_data_shape():
    data = collect_data(10, 3, seed=0)
    assert data.shape == (10, 3)
    # Deterministic with seed
    data2 = collect_data(10, 3, seed=0)
    assert np.array_equal(data, data2)
