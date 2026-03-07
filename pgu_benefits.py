import numpy as np


def calcular_pgu_2026(pension_base_clp: float) -> float:
    """
    Calcula el monto de la Pensión Garantizada Universal (PGU).
    Reglas actualizadas a valores aprox 2026.
    """
    pgu_max = 231732  # Monto base para < 82 años
    limite_inf = 789139  # Límite superior para recibir el 100% de la PGU
    limite_sup = 1252602  # Límite de corte (sobre esto no hay PGU)

    if pension_base_clp <= limite_inf:
        return pgu_max
    elif pension_base_clp < limite_sup:
        # Fórmula legal de decaimiento proporcional
        factor = (limite_sup - pension_base_clp) / (limite_sup - limite_inf)
        return pgu_max * factor
    else:
        return 0.0


# Vectorizamos la función para que Numpy pueda aplicarla a miles de simulaciones a la vez
aplicar_pgu_vectorizado = np.vectorize(calcular_pgu_2026)