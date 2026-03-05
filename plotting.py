import numpy as np
import matplotlib
matplotlib.use("Agg") # Para entornos sin interfaz gráfica (ej. servidores)
import matplotlib.pyplot as plt
import os

def plot_simulated_paths(paths: np.ndarray, asset_index: int = 0, asset_name: str = "Renta Variable", output_file: str = "plot1_simulacion.png"):
    """Grafica múltiples trayectorias para un activo específico."""
    n_sims, n_periods, n_assets = paths.shape
    plt.figure(figsize=(10, 5))
    
    # Mostrar solo un subconjunto de simulaciones (max 30) para no saturar el gráfico
    n_plot = min(30, n_sims)
    plt.plot(paths[:n_plot, :, asset_index].T, color='#1f77b4', alpha=0.15)
    
    # Línea promedio
    plt.plot(paths.mean(axis=0)[:, asset_index], color='#0b3b60', linewidth=2.5, label=f'Promedio Esperado ({asset_name})')
    
    plt.title(f'1. Simulación de Escenarios Estocásticos ({asset_name})', fontsize=14)
    plt.xlabel('Meses', fontsize=12)
    plt.ylabel('Valor del Activo', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def plot_dynamic_weights(weights: np.ndarray, asset_names: list, output_file: str = "plot2_ponderaciones.png"):
    """Grafica la evolución de los pesos (lambda_t) a través del tiempo."""
    n_periods, n_assets = weights.shape
    plt.figure(figsize=(10, 5))
    
    # Colores por defecto para hasta 7 activos
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    plt.stackplot(range(n_periods), weights.T, labels=asset_names, colors=colors[:n_assets], alpha=0.8)
    plt.title('2. Evolución Dinámica de Ponderaciones $\lambda_t$ (Agente GAN)', fontsize=14)
    plt.xlabel('Meses', fontsize=12)
    plt.ylabel('Proporción del Portafolio', fontsize=12)
    # Colocar leyenda fuera del gráfico si son muchos activos
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.margins(0, 0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def plot_portfolio_comparison(paths: np.ndarray, gan_weights: np.ndarray, output_file: str = "plot3_rendimiento.png"):
    """Compara el rendimiento acumulado de la GAN vs un portafolio Naive (1/N)."""
    n_sims, n_periods, n_assets = paths.shape
    
    # 1. Calcular retornos de los activos en cada simulación
    asset_returns = np.diff(paths, axis=1) / paths[:, :-1, :]
    
    # 2. Estrategia Naive (1/N constante)
    naive_weights = np.ones((n_periods - 1, n_assets)) / n_assets
    naive_port_returns = np.sum(asset_returns * naive_weights, axis=-1)
    
    naive_port_value = np.zeros((n_sims, n_periods))
    naive_port_value[:, 0] = 100 # Inversión inicial base 100
    for t in range(1, n_periods):
        naive_port_value[:, t] = naive_port_value[:, t-1] * (1 + naive_port_returns[:, t-1])
        
    # 3. Estrategia GAN (Pesos dinámicos)
    # gan_weights debe tener shape (n_periods, n_assets)
    gan_w = gan_weights[:-1, :] 
    gan_port_returns = np.sum(asset_returns * gan_w, axis=-1)
    
    gan_port_value = np.zeros((n_sims, n_periods))
    gan_port_value[:, 0] = 100
    for t in range(1, n_periods):
        gan_port_value[:, t] = gan_port_value[:, t-1] * (1 + gan_port_returns[:, t-1])

    # 4. Generar el gráfico
    plt.figure(figsize=(10, 5))
    
    # Plot GAN
    plt.plot(gan_port_value.mean(axis=0), label='Portafolio GAN (Promedio)', color='#2ca02c', linewidth=2.5)
    plt.fill_between(range(n_periods), np.percentile(gan_port_value, 5, axis=0), np.percentile(gan_port_value, 95, axis=0), color='#2ca02c', alpha=0.15)
    
    # Plot Naive
    plt.plot(naive_port_value.mean(axis=0), label='Portafolio $1/N$ (Promedio)', color='#7f7f7f', linestyle='--', linewidth=2)
    plt.fill_between(range(n_periods), np.percentile(naive_port_value, 5, axis=0), np.percentile(naive_port_value, 95, axis=0), color='#7f7f7f', alpha=0.15)
    
    plt.title('3. Rendimiento Acumulado: GAN vs Estrategia Estática', fontsize=14)
    plt.xlabel('Meses', fontsize=12)
    plt.ylabel('Valor del Portafolio (Base 100)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return gan_port_value, naive_port_value # Retornamos los valores para calcular el riesgo en el siguiente plot

def plot_risk_distribution(gan_port_value: np.ndarray, naive_port_value: np.ndarray, output_file: str = "plot4_riesgo.png"):
    """Muestra la distribución de la riqueza final y el VaR(95)."""
    gan_final = gan_port_value[:, -1]
    naive_final = naive_port_value[:, -1]

    plt.figure(figsize=(10, 5))
    plt.hist(gan_final, bins=40, alpha=0.7, color='#2ca02c', label='Portafolio Optimizado GAN')
    plt.hist(naive_final, bins=40, alpha=0.6, color='#7f7f7f', label='Portafolio Tradicional $1/N$')

    # Value at Risk 95% (percentil 5 de los resultados)
    var95_gan = np.percentile(gan_final, 5)
    plt.axvline(var95_gan, color='#0f520f', linestyle='dashed', linewidth=2, label=f'GAN VaR(95%) = {var95_gan:.0f}')

    var95_naive = np.percentile(naive_final, 5)
    plt.axvline(var95_naive, color='#4d4d4d', linestyle='dashed', linewidth=2, label=f'1/N VaR(95%) = {var95_naive:.0f}')

    plt.title(f'4. Distribución de Riqueza Final y Riesgo de Cola (Mes {gan_port_value.shape[1]})', fontsize=14)
    plt.xlabel('Valor Final del Portafolio', fontsize=12)
    plt.ylabel('Frecuencia (Nº de Simulaciones)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def generate_report_plots(simulated_paths: np.ndarray, optimal_weights: np.ndarray, asset_names: list, output_dir: str = "."):
    """Función principal para generar todos los gráficos desde main.py"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generando Gráfico 1: Simulaciones...")
    plot_simulated_paths(simulated_paths, asset_index=0, asset_name=asset_names[0], output_file=os.path.join(output_dir, "plot1_simulacion.png"))
    
    print("Generando Gráfico 2: Ponderaciones...")
    plot_dynamic_weights(optimal_weights, asset_names, output_file=os.path.join(output_dir, "plot2_ponderaciones.png"))
    
    print("Generando Gráfico 3: Rendimiento de Portafolio...")
    gan_vals, naive_vals = plot_portfolio_comparison(simulated_paths, optimal_weights, output_file
