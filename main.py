import numpy as np
# Aquí van tus otras importaciones:
# from scenario_generation import calibrate_gbm_parameters, simulate_correlated_gbm
# from adversarial_lambda import train_portfolio_generator
from plotting import generate_report_plots # <--- Nueva importación

def main():
    # ... (Tu código de recolección de datos y calibración) ...
    
    # Asumimos que tienes N activos, generamos nombres de prueba:
    # asset_names = ['AFP Capital', 'AFP Cuprum', 'AFP Habitat', 'AFP Modelo', 'AFP Planvital', 'AFP Provida', 'USD/CLP']
    
    # 1. Simulas tus trayectorias
    # simulated_paths = simulate_correlated_gbm(params, S0, periods, n_sims=500)
    
    # 2. Entrenas la red y obtienes tu tensor de pesos de dimensiones (periods, n_assets)
    # optimal_dynamic_weights = train_portfolio_generator(simulated_paths)
    
    # 3. Generamos todos los gráficos en la carpeta 'resultados'
    asset_names = [f"Activo {i+1}" for i in range(optimal_dynamic_weights.shape[1])]
    generate_report_plots(
        simulated_paths=simulated_paths, 
        optimal_weights=optimal_dynamic_weights, 
        asset_names=asset_names,
        output_dir="resultados_simulacion"
    )

if __name__ == '__main__':
    main()
