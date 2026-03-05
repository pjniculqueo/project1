import torch
from torch import nn
import numpy as np

class DynamicWeightGenerator(nn.Module):
    def __init__(self, latent_dim, periods, n_assets):
        super().__init__()
        self.periods = periods
        self.n_assets = n_assets
        
        # Red secuencial simple para capturar la evolución temporal
        self.lstm = nn.LSTM(latent_dim, 32, batch_first=True)
        self.linear = nn.Linear(32, n_assets)
        self.softmax = nn.Softmax(dim=-1) # Asegura que los pesos sumen 1 en cada t

    def forward(self, z):
        # z shape: (batch_size, periods, latent_dim)
        lstm_out, _ = self.lstm(z)
        logits = self.linear(lstm_out)
        weights = self.softmax(logits)
        return weights # shape: (batch_size, periods, n_assets)

def pseudo_sharpe_loss(weights, simulated_returns, risk_free_rate=0.0):
    """
    Calcula una pérdida basada en el negativo del Ratio de Sharpe.
    El optimizador intentará minimizar esto (maximizando el Sharpe).
    """
    # simulated_returns shape: (batch_size, periods, n_assets)
    # weights shape: (batch_size, periods, n_assets)
    
    # Retorno del portafolio en cada instante t
    portfolio_returns = torch.sum(weights * simulated_returns, dim=-1)
    
    mean_return = torch.mean(portfolio_returns, dim=-1)
    std_return = torch.std(portfolio_returns, dim=-1) + 1e-6 # Evitar división por cero
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return -torch.mean(sharpe) # Retornamos el negativo porque PyTorch minimiza

def train_portfolio_generator(simulated_paths, n_epochs=500, latent_dim=8):
    """
    Entrena el Generador para encontrar la trayectoria óptima de pesos (Decision Intelligence).
    """
    # Convertir precios simulados a retornos periódicos para la optimización
    simulated_returns = np.diff(simulated_paths, axis=1) / simulated_paths[:, :-1, :]
    sim_tensor = torch.tensor(simulated_returns, dtype=torch.float32)
    
    batch_size, periods, n_assets = sim_tensor.shape
    gen = DynamicWeightGenerator(latent_dim, periods, n_assets)
    optimizer = torch.optim.Adam(gen.parameters(), lr=0.005)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Ruido latente de entrada (estado base)
        z = torch.randn(batch_size, periods, latent_dim)
        
        # Generar pesos para cada instante de tiempo
        weights = gen(z)
        
        # Optimizar directamente contra el Sharpe Ratio de las simulaciones
        loss = pseudo_sharpe_loss(weights, sim_tensor)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss (Neg Sharpe): {loss.item():.4f}")

    # Retornar la estrategia de pesos promedio generada tras el entrenamiento
    with torch.no_grad():
        z = torch.randn(1, periods, latent_dim)
        optimal_weights = gen(z).squeeze().numpy()
        
    return optimal_weights
