import numpy as np


def generate_glidepath_weights(periods: int, n_assets: int,
                               risk_assets_idx: list, safe_assets_idx: list,
                               start_risk_pct: float = 0.85, end_risk_pct: float = 0.15) -> np.ndarray:
    """
    Genera la trayectoria de pesos para un Fondo Generacional (Target Date Fund).
    A medida que pasa el tiempo (acercándose a la jubilación), disminuye el riesgo
    transfiriendo capital de renta variable a renta fija.
    """
    weights = np.zeros((periods, n_assets))

    # Transición lineal (o curva) del porcentaje de riesgo a través de los años
    risk_pct_t = np.linspace(start_risk_pct, end_risk_pct, periods)
    safe_pct_t = 1.0 - risk_pct_t

    # Repartir equitativamente el porcentaje de riesgo entre los activos riesgosos
    w_risk_per_asset = risk_pct_t / max(1, len(risk_assets_idx))
    w_safe_per_asset = safe_pct_t / max(1, len(safe_assets_idx))

    for t in range(periods):
        for i in risk_assets_idx:
            weights[t, i] = w_risk_per_asset[t]
        for i in safe_assets_idx:
            weights[t, i] = w_safe_per_asset[t]

    return weights