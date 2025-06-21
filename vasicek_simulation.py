import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch is used for the GAN implementation
import torch
from torch import nn

def download_afp_data(start_year=2000, end_year=2020):
    """Download fund A data from a public GitHub mirror."""
    dfs = []
    for year in range(start_year, end_year + 1):
        url = (
            "https://raw.githubusercontent.com/rpmunoz/afp_chile/master/data/"
            f"vcfA{year}-{year}.csv"
        )
        r = requests.get(url)
        if r.status_code != 200:
            print(f"Failed to download data for {year}")
            continue
        df = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',', skiprows=2)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        # Keep only 'Valor Cuota' columns for each AFP
        providers = ['CAPITAL', 'CUPRUM', 'HABITAT', 'MODELO', 'PLANVITAL', 'PROVIDA']
        cuota_cols = [c for c in df.columns if 'Valor Cuota' in c]
        if len(cuota_cols) != 6:
            # Fallback to selecting every second column after Fecha
            cuota_cols = df.columns[1::2][:6]
        df = df[['Fecha'] + list(cuota_cols)]
        df.columns = ['Fecha'] + providers
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def download_exchange_rate():
    """Download USD/CLP exchange rate from GitHub."""
    url = (
        "https://raw.githubusercontent.com/ignacioTapia95/"
        "forecast-usd-clp-exchange-rate/main/data/raw/exchangeRateIATA.csv"
    )
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
    df.columns = ['Fecha', 'USDCLP']
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df

def compute_monthly_returns(df, value_col_prefix=None):
    """Compute monthly percentage returns from daily values."""
    df = df.set_index('Fecha').resample('M').last()
    returns = df.pct_change().dropna()
    return returns

def fit_vasicek(series):
    """Estimate Vasicek parameters using simple OLS."""
    x = series[:-1].values
    y = series[1:].values
    A = np.vstack([x, np.ones(len(x))]).T
    kappa, c = np.linalg.lstsq(A, y, rcond=None)[0]
    theta = c / (1 - kappa)
    residuals = y - (kappa * x + c)
    sigma = residuals.std(ddof=1)
    return float(kappa), float(theta), float(sigma)

def simulate_vasicek_paths(params, corr, periods, dt, n_sims):
    """Simulate correlated Vasicek processes."""
    kappa, theta, sigma = zip(*params)
    kappa = np.array(kappa)
    theta = np.array(theta)
    sigma = np.array(sigma)
    n_factors = len(params)
    L = np.linalg.cholesky(corr)
    paths = np.zeros((n_sims, periods, n_factors))
    paths[:, 0, :] = theta
    for t in range(1, periods):
        z = np.random.normal(size=(n_sims, n_factors))
        z = z @ L.T
        paths[:, t, :] = (
            paths[:, t-1, :] + kappa*(theta - paths[:, t-1, :])*dt + sigma*np.sqrt(dt)*z
        )
    return paths

class WeightGenerator(nn.Module):
    def __init__(self, latent_dim, n_series):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_series),
            nn.Softmax(dim=1),
        )
    def forward(self, z):
        return self.net(z)

class WeightDiscriminator(nn.Module):
    def __init__(self, n_series):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_series, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

def train_gan(returns, n_epochs=1000, latent_dim=8):
    n_series = returns.shape[1]
    gen = WeightGenerator(latent_dim, n_series)
    disc = WeightDiscriminator(n_series)
    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(gen.parameters(), lr=0.01)
    opt_d = torch.optim.Adam(disc.parameters(), lr=0.01)

    real_weights = torch.ones((returns.shape[0], n_series))/n_series
    for epoch in range(n_epochs):
        z = torch.randn(real_weights.size(0), latent_dim)
        fake_weights = gen(z)
        # Train discriminator
        opt_d.zero_grad()
        real_pred = disc(real_weights)
        fake_pred = disc(fake_weights.detach())
        loss_d = criterion(real_pred, torch.ones_like(real_pred)) + \
                 criterion(fake_pred, torch.zeros_like(fake_pred))
        loss_d.backward()
        opt_d.step()
        # Train generator
        opt_g.zero_grad()
        fake_pred = disc(fake_weights)
        loss_g = criterion(fake_pred, torch.ones_like(fake_pred))
        loss_g.backward()
        opt_g.step()
    z = torch.randn(1, latent_dim)
    return gen(z).detach().numpy().flatten()

def main():
    afp = download_afp_data()
    usdclp = download_exchange_rate()

    afp_returns = compute_monthly_returns(afp)
    fx_returns = compute_monthly_returns(usdclp)

    combined = afp_returns.join(fx_returns, how='inner')

    params = [fit_vasicek(combined[col]) for col in combined.columns]
    corr = combined.corr().values
    periods = 35 * 12
    paths = simulate_vasicek_paths(params, corr, periods, dt=1/12, n_sims=1000)

    weights = train_gan(combined)
    portfolio = (paths * weights).sum(axis=2)
    avg_path = portfolio.mean(axis=0)
    worst_path = np.quantile(portfolio, 0.01, axis=0)

    plt.plot(avg_path, label='Promedio')
    plt.plot(worst_path, label='Percentil 99% Peor')
    plt.legend()
    plt.xlabel('Mes')
    plt.ylabel('Rentabilidad simulada')
    plt.title('Resultado de simulaciones')
    plt.tight_layout()
    plt.savefig('simulacion.png')

if __name__ == '__main__':
    main()
