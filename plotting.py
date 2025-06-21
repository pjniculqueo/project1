import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def plot_kpis(scenarios: np.ndarray, lambda_t: np.ndarray, output_file: str = 'results.png') -> tuple[float, float]:
    """Plot KPIs and compute VaR(99) and average."""
    aggregated = scenarios.sum(axis=1)
    var99 = np.percentile(aggregated, 1)
    avg = aggregated.mean()
    plt.figure()
    plt.hist(aggregated, bins=20, alpha=0.7, label='Aggregated Returns')
    plt.axvline(var99, color='r', linestyle='--', label=f'VaR(99)={var99:.2f}')
    plt.axvline(avg, color='g', linestyle='--', label=f'Avg={avg:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return var99, avg

import os
if __name__ == "__main__":
    from scenario_generation import build_scenarios
    params = {"mu": np.zeros(1), "cov": np.eye(1)}
    scenarios = build_scenarios(params, 100, seed=0)
    lambdas = np.ones(len(scenarios))
    plot_kpis(scenarios, lambdas)
    print(os.path.exists('results.png'))
