import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Importamos las funciones unificadas del ecosistema
from glidepath import generate_glidepath_weights
from retirement_analysis import calcular_pension_mensual
from pgu_benefits import aplicar_pgu_vectorizado


def run_pgu_optimization_analysis(simulated_paths: np.ndarray, output_dir: str = "."):
    """
    Realiza un A/B testing estricto usando las mismas simulaciones de mercado (simulated_paths)
    que usa el resto del proyecto, garantizando consistencia causal.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extraer las mismas tasas de retorno del escenario base
    n_sims, n_periods, n_assets = simulated_paths.shape
    asset_returns = np.diff(simulated_paths, axis=1) / simulated_paths[:, :-1, :]
    periods_returns = n_periods - 1  # Ej: 419 meses

    # Parámetros base
    meses_sobrevida = 240
    risk_indices = [0, 1, 2, 3]
    safe_indices = [4, 5, 6]

    # 2. Construir los Glidepaths con la función centralizada
    w_standard = generate_glidepath_weights(periods_returns, n_assets, risk_indices, safe_indices, 0.85, 0.15)

    # Optimizaciones del Estado
    w_opt_low = generate_glidepath_weights(periods_returns, n_assets, risk_indices, safe_indices, 0.85, 0.85)
    w_opt_med = generate_glidepath_weights(periods_returns, n_assets, risk_indices, safe_indices, 0.85, 0.50)
    w_opt_high = generate_glidepath_weights(periods_returns, n_assets, risk_indices, safe_indices, 0.85, 0.0)

    # 3. Definir Perfiles (Grupos de Tratamiento)
    perfiles = [
        {"name": "Bajo Ingreso\n(Aporte $40k)", "aporte": 40000, "w_opt": w_opt_low},
        {"name": "Medio Ingreso\n(Aporte $100k)", "aporte": 100000, "w_opt": w_opt_med},
        {"name": "Alto Ingreso\n(Aporte $220k)", "aporte": 220000, "w_opt": w_opt_high}
    ]

    gasto_estandar, gasto_optimo, pension_estandar, pension_optima = [], [], [], []

    for p in perfiles:
        # A/B TEST - GRUPO CONTROL (Fondo Generacional Estándar)
        pen_std = calcular_pension_mensual(asset_returns, w_standard, periods_returns, p["aporte"], meses_sobrevida)
        pgu_std = aplicar_pgu_vectorizado(pen_std)
        gasto_estandar.append(pgu_std.mean())
        pension_estandar.append((pen_std + pgu_std).mean())

        # A/B TEST - GRUPO TRATAMIENTO (Riesgo Optimizado)
        pen_opt = calcular_pension_mensual(asset_returns, p["w_opt"], periods_returns, p["aporte"], meses_sobrevida)
        pgu_opt = aplicar_pgu_vectorizado(pen_opt)
        gasto_optimo.append(pgu_opt.mean())
        pension_optima.append((pen_opt + pgu_opt).mean())

    # 4. Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x, width = np.arange(len(perfiles)), 0.35

    # Plot 1: Gasto Fiscal
    ax1.bar(x - width / 2, gasto_estandar, width, label='Fondo Generacional Estándar', color='#7f7f7f', alpha=0.8)
    ax1.bar(x + width / 2, gasto_optimo, width, label='Riesgo Diferenciado Optimizado', color='#d62728', alpha=0.8)
    ax1.set_ylabel('Gasto PGU por Afiliado (CLP)', fontsize=12)
    ax1.set_title('7A. Impacto en el Gasto Fiscal', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([p["name"] for p in perfiles])
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Pensión Final
    ax2.bar(x - width / 2, pension_estandar, width, label='Fondo Generacional Estándar', color='#7f7f7f', alpha=0.8)
    ax2.bar(x + width / 2, pension_optima, width, label='Riesgo Diferenciado Optimizado', color='#1f77b4', alpha=0.8)
    ax2.set_ylabel('Pensión Total (Base + PGU) (CLP)', fontsize=12)
    ax2.set_title('7B. Impacto en la Pensión Total del Afiliado', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([p["name"] for p in perfiles])
    ax2.legend()
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'plot7_optimizacion_riesgo_pgu.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Análisis A/B de Optimización PGU completado. Guardado en: {output_path}")