from data_collection import collect_data
from parameter_calibration import calibrate_parameters
from scenario_generation import build_scenarios
from adversarial_lambda import estimate_lambda
from plotting import plot_kpis


def run_all(num_scenarios: int = 100) -> tuple[float, float]:
    """Run the entire pipeline and return VaR and average."""
    data = collect_data()
    params = calibrate_parameters(data)
    scenarios = build_scenarios(params, num_scenarios)
    lambda_t = estimate_lambda(scenarios)
    return plot_kpis(scenarios, lambda_t)


if __name__ == "__main__":
    var99, avg = run_all()
    print(f"VaR(99): {var99:.4f}  Avg: {avg:.4f}")
