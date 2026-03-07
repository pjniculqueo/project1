import numpy as np
from scenario_generation import calibrate_gbm_parameters, simulate_correlated_gbm
from adversarial_lambda import train_portfolio_generator
from glidepath import generate_glidepath_weights
from plotting import generate_report_plots
from retirement_analysis import run_retirement_analysis
# --- NUEVA IMPORTACIÓN ---
from pgu_optimization_analysis import run_pgu_optimization_analysis


def main():
    print("1. Generando datos históricos para calibración...")
    # 7 activos: Asumiremos que los primeros 4 son Riesgosos y los 3 últimos Seguros
    data = np.random.normal(0.005, 0.02, size=(120, 7))
    S0 = np.ones(7) * 100

    print("2. Calibrando parámetros y simulando trayectorias (GBM)...")
    params = calibrate_gbm_parameters(data)
    periods = 35 * 12  # 420 meses (35 años)
    simulated_paths = simulate_correlated_gbm(params, S0, periods, n_sims=500)

    print("3A. Entrenando Agente GAN (Viejo sistema optimizado)...")
    optimal_gan_weights = train_portfolio_generator(simulated_paths)

    print("3B. Calculando Glidepath (Nuevo sistema de Fondos Generacionales)...")
    risk_indices = [0, 1, 2, 3]
    safe_indices = [4, 5, 6]
    tdf_weights = generate_glidepath_weights(periods, n_assets=7,
                                             risk_assets_idx=risk_indices,
                                             safe_assets_idx=safe_indices,
                                             start_risk_pct=0.85,
                                             end_risk_pct=0.15)

    print("4. Generando y guardando gráficos comparativos (1 al 5)...")
    asset_names = [f"Activo {i + 1} (Riesgo)" if i in risk_indices else f"Activo {i + 1} (Seguro)" for i in range(7)]

    generate_report_plots(
        simulated_paths=simulated_paths,
        gan_weights=optimal_gan_weights,
        tdf_weights=tdf_weights,
        asset_names=asset_names,
        output_dir="resultados_simulacion"
    )

    print("5. Ejecutando análisis de sensibilización por edad (Gráfico 6)...")
    run_retirement_analysis(output_dir="resultados_simulacion")

    print("6. Ejecutando análisis de optimización de riesgo PGU (Gráfico 7)...")
    run_pgu_optimization_analysis(simulated_paths=simulated_paths, output_dir="resultados_simulacion")

    # --- NUEVA EJECUCIÓN ---
    print("✅ Todo el pipeline pericial ha finalizado con éxito.")


if __name__ == '__main__':
    main()