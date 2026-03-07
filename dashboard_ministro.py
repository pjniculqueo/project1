import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Configuración de la página ---
st.set_page_config(page_title="Simulador de Política Pública - Pensiones", layout="wide")
st.title("🏛️ Simulador de Política Pública: Pensiones y Gasto Fiscal")
st.markdown(
    "Herramienta interactiva para evaluar el impacto de distintas regulaciones en el bienestar del pensionado y la carga fiscal del Estado (PGU).")

# --- Barra lateral para parámetros de Política Pública ---
st.sidebar.header("Parámetros del Ciudadano")
edad_jubilacion = st.sidebar.slider("Edad de Jubilación (Años)", min_value=60, max_value=75, value=65, step=1)
aporte_mensual = st.sidebar.slider("Aporte Mensual (CLP)", min_value=20000, max_value=300000, value=80000, step=10000,
                                   help="Define el estrato socioeconómico.")

# Supuestos fijos para simplificar la vista ejecutiva
edad_inicio = 25
meses_sobrevida = 240  # 20 años de retiro


# --- Simulador Matemático ---
@st.cache_data
def simular_mercado(meses_cotizacion, n_sims=1000):
    np.random.seed(42)
    # 7 activos: 4 Riesgosos, 3 Seguros
    mu_anual = np.array([0.08, 0.085, 0.075, 0.09, 0.04, 0.035, 0.045])
    sigma_anual = np.array([0.15, 0.16, 0.14, 0.18, 0.04, 0.03, 0.05])
    return np.random.normal(mu_anual / 12, sigma_anual / np.sqrt(12), size=(n_sims, meses_cotizacion, 7))


@st.cache_data
def calcular_pgu(pensiones_base):
    pgu_max, limite_inf, limite_sup = 231732, 789139, 1252602
    pgu = np.zeros_like(pensiones_base)
    pgu[pensiones_base <= limite_inf] = pgu_max
    mask = (pensiones_base > limite_inf) & (pensiones_base < limite_sup)
    pgu[mask] = pgu_max * ((limite_sup - pensiones_base[mask]) / (limite_sup - limite_inf))
    return pgu


def get_weights(periods, start_r, end_r):
    weights = np.zeros((periods, 7))
    risk_pct = np.linspace(start_r, end_r, periods)
    safe_pct = 1.0 - risk_pct
    for i in range(4): weights[:, i] = risk_pct / 4
    for i in range(4, 7): weights[:, i] = safe_pct / 3
    return weights


def calcular_pension(returns, weights, aporte):
    port_returns = np.sum(returns * weights, axis=-1)
    wealth = np.zeros((returns.shape[0], returns.shape[1]))
    wealth[:, 0] = aporte
    for t in range(1, returns.shape[1]):
        wealth[:, t] = (wealth[:, t - 1] + aporte) * (1 + port_returns[:, t])
    return wealth[:, -1] / meses_sobrevida


# --- Ejecución ---
meses_cotizacion = (edad_jubilacion - edad_inicio) * 12
returns = simular_mercado(meses_cotizacion)

# 1. Política Antigua (Estrategia Activa / Alto Riesgo Sostenido)
w_antigua = get_weights(meses_cotizacion, 0.85, 0.85)

# 2. Nueva Ley (Fondo Generacional Estándar)
w_ley = get_weights(meses_cotizacion, 0.85, 0.15)

# 3. Alternativa Propuesta (Riesgo Optimizado según estrato)
# Si es vulnerable, alto riesgo para que el mercado lo ayude (Estado lo cubre si cae).
# Si es rico, riesgo 0% al final para asegurar su plata y que no caiga a PGU.
if aporte_mensual <= 60000:
    end_risk_alt = 0.85
    desc_alt = "Estrategia Agresiva (Cubre Estado)"
elif aporte_mensual <= 150000:
    end_risk_alt = 0.50
    desc_alt = "Estrategia Moderada"
else:
    end_risk_alt = 0.0
    desc_alt = "Estrategia Conservadora Pura"

w_alt = get_weights(meses_cotizacion, 0.85, end_risk_alt)

# Cálculos
pen_base_antigua = calcular_pension(returns, w_antigua, aporte_mensual)
pen_base_ley = calcular_pension(returns, w_ley, aporte_mensual)
pen_base_alt = calcular_pension(returns, w_alt, aporte_mensual)

pgu_antigua = calcular_pgu(pen_base_antigua)
pgu_ley = calcular_pgu(pen_base_ley)
pgu_alt = calcular_pgu(pen_base_alt)

pen_tot_antigua = pen_base_antigua + pgu_antigua
pen_tot_ley = pen_base_ley + pgu_ley
pen_tot_alt = pen_base_alt + pgu_alt

# --- Dashboard UI ---
st.markdown(
    f"**Escenario evaluado:** Jubilación a los **{edad_jubilacion} años** con aporte mensual de **${aporte_mensual:,.0f}**.")

col1, col2, col3 = st.columns(3)
col1.info(
    f"**1. Sistema Antiguo**\n\nPensión Promedio: **${pen_tot_antigua.mean():,.0f}**\nGasto Fiscal: **${pgu_antigua.mean():,.0f}**")
col2.warning(
    f"**2. Ley Actual (TDF)**\n\nPensión Promedio: **${pen_tot_ley.mean():,.0f}**\nGasto Fiscal: **${pgu_ley.mean():,.0f}**")
col3.success(
    f"**3. Alternativa Propuesta** ({desc_alt})\n\nPensión Promedio: **${pen_tot_alt.mean():,.0f}**\nGasto Fiscal: **${pgu_alt.mean():,.0f}**")

st.markdown("---")

# Gráficos de Contraste Político
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("### Costo Fiscal Promedio por Jubilado")
    fig, ax = plt.subplots(figsize=(6, 4))
    politicas = ['Sistema Antiguo', 'Ley Actual (TDF)', 'Alternativa Propuesta']
    gastos = [pgu_antigua.mean(), pgu_ley.mean(), pgu_alt.mean()]
    colores = ['#7f7f7f', '#d62728', '#2ca02c']

    bars = ax.bar(politicas, gastos, color=colores, alpha=0.8)
    ax.set_ylabel("Gasto PGU Mensual (CLP)")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))

    # Textos sobre barras
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 2000, f"${yval:,.0f}", ha='center', va='bottom', fontsize=10)

    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

with col_g2:
    st.markdown("### Pensión Promedio vs Riesgo (VaR 95%)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    pensiones = [pen_tot_antigua.mean(), pen_tot_ley.mean(), pen_tot_alt.mean()]
    var_95 = [np.percentile(pen_tot_antigua, 5), np.percentile(pen_tot_ley, 5), np.percentile(pen_tot_alt, 5)]

    # Barra de pensión promedio
    bars2 = ax2.bar(politicas, pensiones, color=colores, alpha=0.8, label="Pensión Promedio")
    # Línea de error mostrando el VaR (riesgo de caída)
    yerr = [np.array(pensiones) - np.array(var_95), [0, 0, 0]]
    ax2.errorbar(politicas, pensiones, yerr=yerr, fmt='none', ecolor='black', capsize=5,
                 label="Caída en Crisis (VaR 95%)")

    ax2.set_ylabel("Pensión Total Mensual (CLP)")
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax2.legend()

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 5000, f"${yval:,.0f}", ha='center', va='bottom', fontsize=10)

    ax2.grid(axis='y', alpha=0.3)
    st.pyplot(fig2)

st.markdown("""
**📝 Resumen para el Ministro:**
Mueva los controles de la izquierda. Notará que la **Ley Actual (Roja)** es consistentemente la que entrega peores pensiones a los ciudadanos y, en los estratos medios y bajos, **no le genera un ahorro significativo al Estado** frente a la Alternativa (Verde). La Alternativa adapta el riesgo a la persona, maximizando su bienestar aprovechando la cobertura que ya entrega la PGU.
""")