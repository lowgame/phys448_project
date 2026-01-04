#!/usr/bin/env python3
"""
Tasks 5, 6, 7, 8: CLT Analysis, Price Estimation, Discussion, Extra Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import CONSTS
from helper import helper


def load_data() -> pd.DataFrame:
    """Load the main data.csv file."""
    return pd.read_csv("data.csv", parse_dates=['date'])


def get_price_series(df: pd.DataFrame, symbol: str, price_col: str = 'price') -> pd.Series:
    """Extract price series for a given symbol from the dataframe."""
    symbol_df = df[df['symbol'] == symbol].set_index('date').sort_index()
    return symbol_df[price_col]


def get_change_series(df: pd.DataFrame, symbol: str, change_col: str) -> pd.Series:
    """Extract change series for a given symbol from the dataframe."""
    symbol_df = df[df['symbol'] == symbol].set_index('date').sort_index()
    return symbol_df[change_col].dropna()


# =============================================================================
# TASK 5: Central Limit Theorem Analysis
# =============================================================================

def test_normality(data: np.ndarray, name: str) -> dict:
    """Test normality using Shapiro-Wilk test and compute descriptive stats."""
    # Shapiro-Wilk test (use sample if too large)
    sample_size = min(5000, len(data))
    sample = np.random.choice(data, sample_size, replace=False) if len(data) > 5000 else data
    
    stat, p_value = stats.shapiro(sample)
    
    return {
        'name': name,
        'n': len(data),
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'shapiro_stat': stat,
        'shapiro_p': p_value,
        'is_normal': p_value > 0.05
    }


def demonstrate_clt(data: np.ndarray, name: str, sample_sizes: list = [5, 10, 30, 50, 100]) -> dict:
    """
    Demonstrate CLT by showing that sample means converge to normal distribution.
    """
    n_simulations = 1000
    results = {}
    
    for n in sample_sizes:
        if n > len(data):
            continue
        sample_means = []
        for _ in range(n_simulations):
            sample = np.random.choice(data, n, replace=True)
            sample_means.append(np.mean(sample))
        
        sample_means = np.array(sample_means)
        stat, p_value = stats.shapiro(sample_means[:5000] if len(sample_means) > 5000 else sample_means)
        
        results[n] = {
            'mean_of_means': np.mean(sample_means),
            'std_of_means': np.std(sample_means, ddof=1),
            'theoretical_se': np.std(data, ddof=1) / np.sqrt(n),
            'shapiro_stat': stat,
            'shapiro_p': p_value,
            'is_normal': p_value > 0.05,
            'sample_means': sample_means
        }
    
    return results


def plot_clt_demonstration(df: pd.DataFrame) -> None:
    """Plot CLT demonstration for daily returns."""
    if not os.path.exists("img/task_5"):
        os.makedirs("img/task_5", exist_ok=True)
    
    stock_symbols = list(CONSTS.STOCKS.keys())
    
    # Use the first stock for demonstration
    symbol = stock_symbols[0]
    returns = get_change_series(df, symbol, 'change_in_prcntg_try').values
    
    sample_sizes = [5, 10, 30, 50, 100]
    clt_results = demonstrate_clt(returns, symbol, sample_sizes)
    
    # 1. Original distribution
    plt.figure(figsize=(8, 6))
    plt.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue')
    plt.title(f'Original Distribution: {symbol} Daily Returns')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"img/task_5/clt_original_{symbol.replace('.IS', '')}.png")
    plt.close()
    
    # 2. Sample mean distributions
    for n in sample_sizes:
        if n not in clt_results:
            continue
        
        plt.figure(figsize=(8, 6))
        sample_means = clt_results[n]['sample_means']
        plt.hist(sample_means, bins=50, density=True, alpha=0.7, color='steelblue', label='Sample Means')
        
        # Overlay normal distribution
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        mu = clt_results[n]['mean_of_means']
        sigma = clt_results[n]['std_of_means']
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        
        is_normal = "Yes" if clt_results[n]['is_normal'] else "No"
        plt.title(f'CLT: Sample Size n={n} ({symbol})\nNormal? {is_normal} (p={clt_results[n]["shapiro_p"]:.4f})')
        plt.xlabel('Sample Mean')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"img/task_5/clt_n{n}_{symbol.replace('.IS', '')}.png")
        plt.close()
    
    print(f"CLT demonstration plots saved to img/task_5/")


def plot_normality_tests(df: pd.DataFrame) -> dict:
    """Plot histograms and Q-Q plots for daily returns, saved separately for each symbol."""
    if not os.path.exists("img/task_5"):
        os.makedirs("img/task_5", exist_ok=True)
    
    stock_symbols = list(CONSTS.STOCKS.keys())
    all_symbols = stock_symbols + [CONSTS.CURRENCY]
    
    normality_results = []
    
    for symbol in all_symbols:
        returns = get_change_series(df, symbol, 'change_in_prcntg_try').values
        
        # Test normality
        result = test_normality(returns, symbol)
        normality_results.append(result)
        
        # Create a figure with 2 subplots for this symbol
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with normal overlay
        ax1.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', label='Data')
        x = np.linspace(returns.min(), returns.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, result['mean'], result['std']), 'r-', linewidth=2, label='Normal')
        ax1.set_title(f'{symbol}: Histogram of Daily Returns')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title(f'{symbol}: Q-Q Plot (Shapiro p={result["shapiro_p"]:.4f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"img/task_5/normality_test_{symbol.replace('.IS', '').replace('=X', '')}.png")
        plt.close()
    
    print(f"Normality test plots saved to img/task_5/")
    
    return normality_results


def plot_histograms_only(df: pd.DataFrame) -> None:
    """Plot only histograms with normal fits for each stock and USD/TRY, saved separately."""
    if not os.path.exists("img/task_5"):
        os.makedirs("img/task_5", exist_ok=True)
    
    stock_symbols = list(CONSTS.STOCKS.keys())
    all_symbols = stock_symbols + [CONSTS.CURRENCY]
    
    for symbol in all_symbols:
        returns = get_change_series(df, symbol, 'change_in_prcntg_try').values
        
        # Calculate statistics
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Shapiro-Wilk test
        sample_size = min(5000, len(returns))
        sample = np.random.choice(returns, sample_size, replace=False) if len(returns) > 5000 else returns
        shapiro_stat, p_value = stats.shapiro(sample)
        
        plt.figure(figsize=(10, 7))
        
        # Histogram
        plt.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', label='Actual Data')
        
        # Normal fit overlay
        x = np.linspace(returns.min(), returns.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='Normal Fit')
        
        # Add vertical line at mean
        plt.axvline(mean, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        
        plt.title(f'Daily Returns: {symbol.replace(".IS", "").replace("=X", "")}', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        
        # Add text box with statistics
        textstr = (f'μ = {mean:.4f}\n'
                   f'σ = {std:.4f}\n'
                   f'Skew = {skewness:.3f}\n'
                   f'Kurt = {kurtosis:.3f}\n'
                   f'Shapiro W = {shapiro_stat:.4f}\n'
                   f'p-value = {p_value:.4e}')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.gca().text(0.97, 0.97, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"img/task_5/histogram_{symbol.replace('.IS', '').replace('=X', '')}.png", dpi=150)
        plt.close()
    
    print(f"Individual histograms saved to img/task_5/")


# =============================================================================
# TASK 6: Price Estimation Using Correlation
# =============================================================================

def find_most_correlated_pair_usd(df: pd.DataFrame) -> tuple:
    """Find the most highly correlated pair of stocks in USD."""
    stock_symbols = list(CONSTS.STOCKS.keys())
    
    max_corr = -1
    best_pair = None
    
    for i in range(len(stock_symbols)):
        for j in range(i + 1, len(stock_symbols)):
            changes_i = get_change_series(df, stock_symbols[i], 'change_in_prcntg_usd')
            changes_j = get_change_series(df, stock_symbols[j], 'change_in_prcntg_usd')
            
            # Align
            aligned = pd.concat([changes_i, changes_j], axis=1).dropna()
            corr = helper.calculate_correlation(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
            
            if corr > max_corr:
                max_corr = corr
                best_pair = (stock_symbols[i], stock_symbols[j])
    
    return best_pair, max_corr


def estimate_price_change(df: pd.DataFrame, stock_x: str, stock_y: str) -> dict:
    """
    Estimate price change of stock_x using stock_y data.
    Formula: E[ΔX/σ_X] ≈ ρ(X,Y) * (ΔY/σ_Y)
    Therefore: E[ΔX] ≈ ρ(X,Y) * σ_X * (ΔY/σ_Y)
    """
    # Get USD price changes (percentage)
    changes_x = get_change_series(df, stock_x, 'change_in_prcntg_usd')
    changes_y = get_change_series(df, stock_y, 'change_in_prcntg_usd')
    
    # Align data
    aligned = pd.concat([changes_x, changes_y], axis=1).dropna()
    aligned.columns = [stock_x, stock_y]
    
    x_data = aligned[stock_x].values
    y_data = aligned[stock_y].values
    
    # Calculate statistics
    rho = helper.calculate_correlation(x_data, y_data)
    sigma_x = helper.calculate_std_dev(x_data)
    sigma_y = helper.calculate_std_dev(y_data)
    
    # Estimate X changes from Y changes
    # E[ΔX/σ_X] ≈ ρ(X,Y) * (ΔY/σ_Y)
    # E[ΔX] ≈ ρ(X,Y) * σ_X * (ΔY/σ_Y)
    estimated_x_normalized = rho * (y_data / sigma_y)
    estimated_x = estimated_x_normalized * sigma_x
    
    # Get actual USD prices for cumulative comparison
    prices_x = get_price_series(df, stock_x, 'price_usd')
    prices_y = get_price_series(df, stock_y, 'price_usd')
    
    # Calculate total price change 2021-now
    actual_total_change_x = (prices_x.iloc[-1] - prices_x.iloc[0]) / prices_x.iloc[0]
    actual_total_change_y = (prices_y.iloc[-1] - prices_y.iloc[0]) / prices_y.iloc[0]
    
    # Estimate total change using the formula
    # For total change: E[Total ΔX] ≈ ρ * σ_X * (Total ΔY / σ_Y)
    estimated_total_change_x = rho * (sigma_x / sigma_y) * actual_total_change_y
    
    # Calculate prediction errors
    daily_mse = np.mean((x_data - estimated_x) ** 2)
    daily_mae = np.mean(np.abs(x_data - estimated_x))
    daily_r_squared = 1 - np.sum((x_data - estimated_x) ** 2) / np.sum((x_data - np.mean(x_data)) ** 2)
    
    return {
        'stock_x': stock_x,
        'stock_y': stock_y,
        'correlation': rho,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'actual_total_change_x': actual_total_change_x,
        'actual_total_change_y': actual_total_change_y,
        'estimated_total_change_x': estimated_total_change_x,
        'estimation_error': abs(actual_total_change_x - estimated_total_change_x),
        'daily_mse': daily_mse,
        'daily_mae': daily_mae,
        'daily_r_squared': daily_r_squared,
        'actual_daily': x_data,
        'estimated_daily': estimated_x,
        'dates': aligned.index
    }


def plot_price_estimation(result: dict) -> None:
    """Plot actual vs estimated price changes."""
    if not os.path.exists("img"):
        os.makedirs("img")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot: actual vs estimated
    ax1 = axes[0, 0]
    ax1.scatter(result['actual_daily'], result['estimated_daily'], alpha=0.5, s=10)
    ax1.plot([result['actual_daily'].min(), result['actual_daily'].max()], 
             [result['actual_daily'].min(), result['actual_daily'].max()], 
             'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel(f'Actual {result["stock_x"]} Daily Change')
    ax1.set_ylabel(f'Estimated {result["stock_x"]} Daily Change')
    ax1.set_title(f'Actual vs Estimated Daily Changes\nR² = {result["daily_r_squared"]:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series comparison
    ax2 = axes[0, 1]
    ax2.plot(result['dates'], result['actual_daily'], alpha=0.7, label='Actual', linewidth=0.8)
    ax2.plot(result['dates'], result['estimated_daily'], alpha=0.7, label='Estimated', linewidth=0.8)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Change')
    ax2.set_title(f'{result["stock_x"]} Daily Changes: Actual vs Estimated from {result["stock_y"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative returns comparison
    ax3 = axes[1, 0]
    cum_actual = np.cumprod(1 + result['actual_daily']) - 1
    cum_estimated = np.cumprod(1 + result['estimated_daily']) - 1
    ax3.plot(result['dates'], cum_actual, label='Actual Cumulative', linewidth=1.5)
    ax3.plot(result['dates'], cum_estimated, label='Estimated Cumulative', linewidth=1.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative Return')
    ax3.set_title('Cumulative Returns: Actual vs Estimated')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = axes[1, 1]
    errors = result['actual_daily'] - result['estimated_daily']
    ax4.hist(errors, bins=50, density=True, alpha=0.7, color='steelblue')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Error Distribution\nMAE = {result["daily_mae"]:.6f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("img/task_6_price_estimation.png")
    plt.close()
    print("Price estimation plot saved to img/task_6_price_estimation.png")

# =============================================================================
# TASK 8: Extra Analysis
# =============================================================================

def calculate_rolling_correlation(df: pd.DataFrame, symbol1: str, symbol2: str, 
                                   window: int = 60, change_col: str = 'change_in_prcntg_usd') -> pd.Series:
    """Calculate rolling correlation between two symbols."""
    changes1 = get_change_series(df, symbol1, change_col)
    changes2 = get_change_series(df, symbol2, change_col)
    
    aligned = pd.concat([changes1, changes2], axis=1).dropna()
    aligned.columns = [symbol1, symbol2]
    
    rolling_corr = aligned[symbol1].rolling(window=window).corr(aligned[symbol2])
    return rolling_corr


def plot_rolling_correlations(df: pd.DataFrame) -> None:
    """Plot rolling correlations over time."""
    if not os.path.exists("img"):
        os.makedirs("img")
    
    stock_symbols = list(CONSTS.STOCKS.keys())
    windows = [30, 60, 90]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Banking sector rolling correlation
    ax = axes[0]
    for window in windows:
        rolling = calculate_rolling_correlation(df, 'GARAN.IS', 'AKBNK.IS', window)
        ax.plot(rolling.index, rolling.values, label=f'{window}-day window', alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Banking Sector: GARAN.IS vs AKBNK.IS Rolling Correlation')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy sector rolling correlation
    ax = axes[1]
    for window in windows:
        rolling = calculate_rolling_correlation(df, 'ENJSA.IS', 'ZOREN.IS', window)
        ax.plot(rolling.index, rolling.values, label=f'{window}-day window', alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Energy Sector: ENJSA.IS vs ZOREN.IS Rolling Correlation')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Food sector rolling correlation
    ax = axes[2]
    for window in windows:
        rolling = calculate_rolling_correlation(df, 'BIMAS.IS', 'ULKER.IS', window)
        ax.plot(rolling.index, rolling.values, label=f'{window}-day window', alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Food Sector: BIMAS.IS vs ULKER.IS Rolling Correlation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("img/task_8_rolling_correlations.png")
    plt.close()
    print("Rolling correlations saved to img/task_8_rolling_correlations.png")


def calculate_volatility(df: pd.DataFrame, symbol: str, window: int = 21) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns)."""
    changes = get_change_series(df, symbol, 'change_in_prcntg_usd')
    return changes.rolling(window=window).std() * np.sqrt(252)  # Annualized


def plot_volatility_analysis(df: pd.DataFrame) -> None:
    """Plot volatility analysis for all stocks."""
    if not os.path.exists("img"):
        os.makedirs("img")
    
    stock_symbols = list(CONSTS.STOCKS.keys())
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Volatility over time
    ax = axes[0]
    for symbol in stock_symbols:
        vol = calculate_volatility(df, symbol)
        ax.plot(vol.index, vol.values, label=symbol, alpha=0.7)
    ax.set_title('21-Day Rolling Annualized Volatility (USD Returns)')
    ax.set_ylabel('Volatility')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Volatility distribution
    ax = axes[1]
    volatilities = []
    labels = []
    for symbol in stock_symbols:
        vol = calculate_volatility(df, symbol).dropna()
        volatilities.append(vol.values)
        labels.append(symbol)
    
    ax.boxplot(volatilities, tick_labels=labels)
    ax.set_title('Volatility Distribution by Stock')
    ax.set_ylabel('Annualized Volatility')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("img/task_8_volatility_analysis.png")
    plt.close()
    print("Volatility analysis saved to img/task_8_volatility_analysis.png")


def calculate_beta(df: pd.DataFrame, symbol: str) -> float:
    """Calculate beta relative to an equally-weighted market portfolio."""
    stock_symbols = list(CONSTS.STOCKS.keys())
    
    # Create market portfolio (equal weight)
    market_returns = None
    for s in stock_symbols:
        changes = get_change_series(df, s, 'change_in_prcntg_usd')
        if market_returns is None:
            market_returns = changes / len(stock_symbols)
        else:
            aligned = pd.concat([market_returns, changes], axis=1).dropna()
            market_returns = aligned.iloc[:, 0] + aligned.iloc[:, 1] / len(stock_symbols)
    
    stock_returns = get_change_series(df, symbol, 'change_in_prcntg_usd')
    
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ['stock', 'market']
    
    cov = helper.calculate_covariance(aligned['stock'].values, aligned['market'].values)
    var_market = helper.calculate_variance(aligned['market'].values)
    
    return cov / var_market

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load data
    df = load_data()
    print(f"Loaded data.csv with {len(df)} rows.")

    # Call the function you want.

    print("Images saved to img/ directory.")


if __name__ == "__main__":
    main()
