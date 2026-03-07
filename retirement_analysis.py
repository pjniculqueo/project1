import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def simular_retornos_simples(n_sims: int, periods: int, n_assets: int) -> np.ndarray:
    """
    Genera retornos mensuales simulados.
    Para este análisis usamos una aproximación normal mensualizada.
    """
    # Parámetros anualizados convertidos a mensuales
    mu_anual = np.array([0.08, 0.085, 0.075, 0.09, 0.04, 0.035, 0.045])
    sigma_anual = np.array([0.15, 0.16, 0.14, 0.18, 0.04, 0.03, 0.05])

    mu_m = mu_anual / 12
    sigma_m = sigma_anual / np.sqrt(12)

    return np.random.normal(mu_m, sigma_m, size=(n_sims, periods, n_assets))


def calcular_tdf_weights(periods: int, n_assets: int, risk_indices: list, safe_indices: list) -> np.ndarray:
    """Calcula la trayectoria del Fondo Generacional (Glidepath)."""
    weights = np.zeros((periods, n_assets))
    risk_pct_t = np.linspace(0.85, 0.15, periods)
    safe_pct_t = 1.0 - risk_pct_t

    w_risk = risk_pct_t / len(risk_indices)
    w_safe = safe_pct_t / len(safe_indices)

    for i in risk_indices: weights[:, i] = w_risk
    for i in safe_indices: weights[:, i] = w_safe
    return weights


def proxy_gan_weights(periods: int, n_assets: int, risk_indices: list, safe_indices: list) -> np.ndarray:
    """
    Simula el comportamiento de la GAN (Estrategia Activa).
    Mantiene un perfil de alto riesgo (70%-90%) para maximizar retorno a largo plazo.
    """
    weights = np.zeros((periods, n_assets))
    risk_pct_t = np.random.uniform(0.70, 0.90, periods)
    safe_pct_t = 1.0 - risk_pct_t

    w_risk = risk_pct_t / len(risk_indices)
    w_safe = safe_pct_t / len(safe_indices)

    for i in risk_indices: weights[:, i] = w_risk
    for i in safe_indices: weights[:, i] = w_safe
    return weights


def calcular_pension_mensual(returns: np.ndarray, weights: np.ndarray, periods: int, aporte_mensual: float,
                             meses_sobrevida: int) -> np.ndarray:
    """
    Calcula la acumulación de riqueza con aportes mensuales y
    luego la divide por la expectativa de vida en meses para obtener la pensión.
    """
    n_sims = returns.shape[0]
    port_returns = np.sum(returns * weights, axis=-1)

    wealth = np.zeros((n_sims, periods))
    wealth[:, 0] = aporte_mensual

    for t in range(1, periods):
        # El capital del mes anterior más el nuevo aporte, multiplicado por la rentabilidad del mes
        wealth[:, t] = (wealth[:, t - 1] + aporte_mensual) * (1 + port_returns[:, t])

    pension_primer_mes = wealth[:, -1] / meses_sobrevida
    return pension_primer_mes


def run_retirement_analysis(output_dir: str = "."):
    """Función principal que orquesta la simulación y el gráfico."""
    os.makedirs(output_dir, exist_ok=True)

    # Configuración base
    np.random.seed(42)
    n_sims = 500
    n_assets = 7
    risk_indices = [0, 1, 2, 3]
    safe_indices = [4, 5, 6]

    aporte_mensual = 100000
    meses_sobrevida = 240  # 20 años de jubilación

    edades = [60, 65, 70]
    meses_cotizacion = [(e - 25) * 12 for e in edades]  # Asumiendo inicio laboral a los 25

    resultados_gan = []
    resultados_tdf = []
    var_gan = []
    var_tdf = []

    print("Simulando escenarios para diferentes edades de jubilación...")

    for periods in meses_cotizacion:
        ret = simular_retornos_simples(n_sims, periods, n_assets)
        w_tdf = calcular_tdf_weights(periods, n_assets, risk_indices, safe_indices)
        w_gan = proxy_gan_weights(periods, n_assets, risk_indices, safe_indices)

        pen_gan = calcular_pension_mensual(ret, w_gan, periods, aporte_mensual, meses_sobrevida)
        pen_tdf = calcular_pension_mensual(ret, w_tdf, periods, aporte_mensual, meses_sobrevida)

        resultados_gan.append(pen_gan.mean())
        resultados_tdf.append(pen_tdf.mean())

        # Value at Risk 95% (peor escenario)
        var_gan.append(np.percentile(pen_gan, 5))
        var_tdf.append(np.percentile(pen_tdf, 5))

    # --- Creación del Gráfico ---
    print("Generando Gráfico 6: Sensibilización de Edad de Jubilación...")
    x = np.arange(len(edades))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width / 2, resultados_gan, width, label='Antiguo (Estrategia Activa/GAN)', color='#2ca02c',
                    alpha=0.8)
    rects2 = ax.bar(x + width / 2, resultados_tdf, width, label='Nuevo (Fondo Generacional TDF)', color='#d62728',
                    alpha=0.8)

    # Barras de error (VaR 95% hacia abajo)
    ax.errorbar(x - width / 2, resultados_gan, yerr=[np.array(resultados_gan) - np.array(var_gan), np.zeros(3)],
                fmt='none', ecolor='black', capsize=5, label='Riesgo de Cola (VaR 95%)')
    ax.errorbar(x + width / 2, resultados_tdf, yerr=[np.array(resultados_tdf) - np.array(var_tdf), np.zeros(3)],
                fmt='none', ecolor='black', capsize=5)

    ax.set_ylabel('Monto de la Pensión (1er Mes, CLP)', fontsize=12)
    ax.set_title('6. Pensión Estimada según Edad de Jubilación\n(Aporte: $100.000/mes | Ingreso: 25 años)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{e} años\n({m / 12:.0f} años de cotización)' for e, m in zip(edades, meses_cotizacion)])
    ax.legend(loc='upper left')

    # Formatear el eje Y con separador de miles
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))

    # Etiquetas sobre las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'${height:,.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'plot6_edades_jubilacion.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✅ Análisis completado. Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    run_retirement_analysis("resultados_simulacion")