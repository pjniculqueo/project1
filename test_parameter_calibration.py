import numpy as np
from parameter_calibration import calibrate_parameters

def test_calibrate_parameters_output():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    params = calibrate_parameters(data)
    assert "mu" in params and "cov" in params and "corr" in params
    assert params["mu"].shape == (2,)
    assert params["cov"].shape == (2, 2)
    assert params["corr"].shape == (2, 2)
