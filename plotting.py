import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pgu_benefits import aplicar_pgu_vectorizado

def plot_simulated_paths(paths: np.ndarray, asset_index: int = 0, asset_name: str = "Activo",
                         output_file: str = "plot1_simulacion.png"):
    n_sims, n_periods, n_assets = paths.shape
    plt.figure(figsize=(10, 5))
    n_plot = min(30, n_sims)
    plt.plot(paths[:n_plot, :, asset_index].T, color='#1f77b4', alpha=0.15)
    plt.plot(paths.mean(axis=0)[:, asset_index], color='#0b3b60', linewidth=2.5,
             label=f'Promedio Esperado ({asset_name})')
    plt.title(f'1. Simulación de Escenarios Estocásticos ({asset_name})', fontsize=14)
    plt.xlabel('Meses', fontsize=12)
    plt.ylabel('Valor del Activo', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_strategy_weights(weights: np.ndarray, asset_names: list, title: str, output_file: str):
    """Grafica la evolución de pesos para cualquier estrategia (GAN o Glidepath)."""
    n_periods, n_assets = weights.shape
    plt.figure(figsize=(10, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    plt.stackplot(range(n_periods), weights.T, labels=asset_names, colors=colors[:n_assets], alpha=0.8)
    plt.title(title, fontsize=14)
    plt.xlabel('Meses', fontsize=12)
    plt.ylabel('Proporción del Portafolio', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.margins(0, 0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def calculate_portfolio_value(asset_returns, weights, n_sims, n_periods):
    """Calcula el valor acumulado del portafolio dadas las trayectorias y pesos."""
    port_returns = np.sum(asset_returns * weights, axis=-1)
    port_value = np.zeros((n_sims, n_periods))
    port_value[:, 0] = 100
    for t in range(1, n_periods):
        port_value[:, t] = port_value[:, t - 1] * (1 + port_returns[:, t - 1])
    return port_value


def plot_portfolio_comparison(paths: np.ndarray, gan_weights: np.ndarray, tdf_weights: np.ndarray, output_file: str):
    n_sims, n_periods, n_assets = paths.shape
    asset_returns = np.diff(paths, axis=1) / paths[:, :-1, :]

    # 1. Estrategia Naive (1/N estático)
    naive_weights = np.ones((n_periods - 1, n_assets)) / n_assets
    naive_val = calculate_portfolio_value(asset_returns, naive_weights, n_sims, n_periods)

    # 2. Estrategia GAN (Ajuste dinámico optimizado - Antiguo sistema ideal)
    gan_val = calculate_portfolio_value(asset_returns, gan_weights, n_sims, n_periods)

    # 3. Estrategia Fondo Generacional TDF (Glidepath - Nuevo sistema)
    tdf_val = calculate_portfolio_value(asset_returns, tdf_weights[:-1], n_sims, n_periods)

    plt.figure(figsize=(12, 6))

    # Ploteos
    plt.plot(gan_val.mean(axis=0), label='Antiguo: Agente GAN (Promedio)', color='#2ca02c', linewidth=2.5)
    plt.plot(tdf_val.mean(axis=0), label='Nuevo: Fondo Generacional TDF (Promedio)', color='#d62728', linewidth=2.5)
    plt.plot(naive_val.mean(axis=0), label='Basal: Portafolio $1/N$ (Promedio)', color='#7f7f7f', linestyle='--',
             linewidth=2)

    # Intervalos de confianza (solo GAN y TDF para no saturar)
    plt.fill_between(range(n_periods), np.percentile(gan_val, 5, axis=0), np.percentile(gan_val, 95, axis=0),
                     color='#2ca02c', alpha=0.1)
    plt.fill_between(range(n_periods), np.percentile(tdf_val, 5, axis=0), np.percentile(tdf_val, 95, axis=0),
                     color='#d62728', alpha=0.1)

    plt.title('3. Rendimiento Acumulado: GAN vs Nuevo Fondo Generacional', fontsize=14)
    plt.xlabel('Meses', fontsize=12)
    plt.ylabel('Valor del Portafolio (Base 100)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    return gan_val, tdf_val, naive_val


def plot_risk_distribution(gan_val: np.ndarray, tdf_val: np.ndarray, naive_val: np.ndarray, output_file: str):
    gan_final, tdf_final, naive_final = gan_val[:, -1], tdf_val[:, -1], naive_val[:, -1]

    plt.figure(figsize=(10, 6))
    plt.hist(gan_final, bins=40, alpha=0.5, color='#2ca02c', label='Agente GAN')
    plt.hist(tdf_final, bins=40, alpha=0.5, color='#d62728', label='Fondo Generacional TDF')
    plt.hist(naive_final, bins=40, alpha=0.5, color='#7f7f7f', label='Basal 1/N')

    v_gan = np.percentile(gan_final, 5)
    v_tdf = np.percentile(tdf_final, 5)

    plt.axvline(v_gan, color='#0f520f', linestyle='dashed', linewidth=2, label=f'VaR GAN = {v_gan:.0f}')
    plt.axvline(v_tdf, color='#7a1616', linestyle='dashed', linewidth=2, label=f'VaR TDF = {v_tdf:.0f}')

    plt.title(f'4. Distribución Final y Riesgo de Cola (Mes {gan_val.shape[1]})', fontsize=14)
    plt.xlabel('Riqueza Final del Portafolio', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def generate_report_plots(simulated_paths: np.ndarray, gan_weights: np.ndarray, tdf_weights: np.ndarray,
                          asset_names: list, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    plot_simulated_paths(simulated_paths, 0, asset_names[0], os.path.join(output_dir, "plot1_simulacion.png"))

    # Graficamos ambas estrategias de asignación
    plot_strategy_weights(gan_weights, asset_names, r'2A. Evolución Pesos: Inteligencia Artificial (GAN)',
                          os.path.join(output_dir, "plot2A_ponderaciones_gan.png"))
    plot_strategy_weights(tdf_weights, asset_names, r'2B. Evolución Pesos: Fondo Generacional (Glidepath)',
                          os.path.join(output_dir, "plot2B_ponderaciones_tdf.png"))

    # Graficamos la comparativa final
    gan_v, tdf_v, naive_v = plot_portfolio_comparison(simulated_paths, gan_weights, tdf_weights,
                                                      os.path.join(output_dir, "plot3_rendimiento.png"))
    plot_risk_distribution(gan_v, tdf_v, naive_v, os.path.join(output_dir, "plot4_riesgo.png"))
    print(f"✅ Todos los gráficos comparativos fueron guardados en: {os.path.abspath(output_dir)}")
    print("Generando Gráfico 5: Impacto Estatal PGU...")
    plot_impacto_pgu(tdf_v, os.path.join(output_dir, "plot5_impacto_pgu.png"))

def plot_impacto_pgu(tdf_val: np.ndarray, output_file: str = "plot5_impacto_pgu.png"):
    """
    Convierte la riqueza final del Fondo Generacional a pensión mensual
    y grafica el impacto protector de la PGU.
    """
    riqueza_final = tdf_val[:, -1]

    # Supuesto: Convertimos el índice de riqueza base 100 a un capital en CLP.
    # Asumimos que la media de la simulación equivale a una pensión de $600.000
    factor_escala = 600000 / np.median(riqueza_final)
    pension_base_clp = riqueza_final * factor_escala

    # 2. Aplicar subsidio estatal
    subsidio_pgu = aplicar_pgu_vectorizado(pension_base_clp)
    pension_total_clp = pension_base_clp + subsidio_pgu

    plt.figure(figsize=(12, 6))

    # Histogramas
    plt.hist(pension_base_clp, bins=50, range=(0, 2000000), alpha=0.6, color='#7f7f7f',
             label='Pensión Base (Autofinanciada)')
    plt.hist(pension_total_clp, bins=50, range=(0, 2000000), alpha=0.7, color='#1f77b4',
             label='Pensión Total (Base + PGU)')

    var_base = np.percentile(pension_base_clp, 5)
    var_total = np.percentile(pension_total_clp, 5)

    plt.axvline(var_base, color='#4d4d4d', linestyle='dashed', linewidth=2, label=f'VaR(95%) Base = ${var_base:,.0f}')
    plt.axvline(var_total, color='#0b3b60', linestyle='dashed', linewidth=2,
                label=f'VaR(95%) Total = ${var_total:,.0f}')

    plt.axvline(789139, color='red', linestyle='dotted', alpha=0.5, label='Límite PGU Máxima ($789.139)')
    plt.axvline(1252602, color='orange', linestyle='dotted', alpha=0.5, label='Corte PGU ($1.252.602)')

    plt.title('5. Impacto de la PGU en el Riesgo de Ruina del Fondo Generacional', fontsize=14)
    plt.xlabel('Monto de la Pensión Estimada (CLP)', fontsize=12)
    plt.ylabel('Frecuencia (Simulaciones)', fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()