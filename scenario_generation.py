import numpy as np

def calibrate_gbm_parameters(returns: np.ndarray) -> dict:
    """
    Calibra la deriva (mu) y volatilidad (sigma) para un Movimiento Browniano Geométrico.
    Asume que 'returns' son retornos logarítmicos o porcentuales periódicos.
    """
    mu = returns.mean(axis=0)
    sigma = returns.std(axis=0, ddof=1)
    corr_matrix = np.corrcoef(returns, rowvar=False)
    
    # Descomposición de Cholesky para simular la correlación
    L = np.linalg.cholesky(corr_matrix)
    
    return {"mu": mu, "sigma": sigma, "L": L}

def simulate_correlated_gbm(params: dict, S0: np.ndarray, periods: int, n_sims: int, dt: float = 1/12) -> np.ndarray:
    """
    Simula trayectorias correlacionadas usando Movimiento Browniano Geométrico (GBM).
    Ideal para Renta Variable (Fondo A) y Tipo de Cambio (USDCLP).
    
    Retorna: Array de dimensiones (n_sims, periods, n_assets)
    """
    mu = params["mu"]
    sigma = params["sigma"]
    L = params["L"]
    n_assets = len(mu)
    
    # Inicializar el tensor de trayectorias
    paths = np.zeros((n_sims, periods, n_assets))
    paths[:, 0, :] = S0
    
    for t in range(1, periods):
        # Generar ruido blanco normal y aplicar correlación
        z = np.random.standard_normal((n_sims, n_assets))
        z_corr = z @ L.T
        
        # Ecuación discreta del GBM
        paths[:, t, :] = paths[:, t-1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_corr
        )
        
    return paths
