import os
import numpy as np
from plotting import plot_kpis


def test_plot_kpis_creates_file(tmp_path):
    scenarios = np.zeros((10, 2))
    lambdas = np.ones(10)
    output = tmp_path / 'plot.png'
    var99, avg = plot_kpis(scenarios, lambdas, output_file=str(output))
    assert output.exists()
    assert isinstance(var99, float)
    assert isinstance(avg, float)
