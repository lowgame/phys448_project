"""
Task 8: Extra Analysis and Discussion
=====================================
Comprehensive statistical analysis of Turkish stock market data.
Includes: descriptive statistics, risk metrics, sector analysis, 
currency impact, time-series analysis, and advanced statistical tests.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, jarque_bera, anderson
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, levene, bartlett
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add helper module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper.helper import calculate_correlation, calculate_std_dev, calculate_variance

# Constants
STOCKS = ['GARAN.IS', 'AKBNK.IS', 'ENJSA.IS', 'ZOREN.IS', 'BIMAS.IS', 'ULKER.IS']
SECTORS = {
    'Banking': ['GARAN.IS', 'AKBNK.IS'],
    'Energy': ['ENJSA.IS', 'ZOREN.IS'],
    'Food': ['BIMAS.IS', 'ULKER.IS']
}
CURRENCY = 'USDTRY=X'


def load_data():
    """Load and prepare data from CSV."""
    df = pd.read_csv('data.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded data.csv with {len(df)} rows.")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def descriptive_statistics(df):
    """Comprehensive descriptive statistics for each stock."""
    
    all_stats = []
    
    for symbol in STOCKS + [CURRENCY]:
        data = df[df['symbol'] == symbol]
        prices = data['price'].values
        returns = data['change_in_prcntg_try'].dropna().values
        
        stock_stats = {
            'Symbol': symbol,
            'N': len(prices),
            'Mean Price': np.mean(prices),
            'Median Price': np.median(prices),
            'Std Dev': np.std(prices, ddof=1),
            'Variance': np.var(prices, ddof=1),
            'Min': np.min(prices),
            'Max': np.max(prices),
            'Range': np.max(prices) - np.min(prices),
            'IQR': np.percentile(prices, 75) - np.percentile(prices, 25),
            'Skewness': stats.skew(prices),
            'Kurtosis': stats.kurtosis(prices),
            'CV (%)': (np.std(prices, ddof=1) / np.mean(prices)) * 100,
            'Mean Daily Return (%)': np.mean(returns) * 100 if len(returns) > 0 else 0,
            'Volatility (%)': np.std(returns, ddof=1) * 100 if len(returns) > 0 else 0,
        }
        all_stats.append(stock_stats)
    
    
    stats_df = pd.DataFrame(all_stats)
    
    # Interpretation
    
    highest_vol = max(all_stats[:-1], key=lambda x: x['Volatility (%)'])
    lowest_vol = min(all_stats[:-1], key=lambda x: x['Volatility (%)'])
    
    
    return stats_df


def risk_metrics(df):
    """Calculate risk metrics: VaR, CVaR, Maximum Drawdown, Sharpe Ratio."""
    
    risk_data = []
    risk_free_rate = 0.40  # Annual risk-free rate (approx Turkish rate)
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol].sort_values('date')
        returns = data['change_in_prcntg_try'].dropna().values
        prices = data['price'].values
        
        # Value at Risk (VaR) - Historical method
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Sharpe Ratio (annualized)
        excess_returns = returns - daily_rf
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        
        # Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0
        sortino = np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio
        annual_return = (1 + np.mean(returns)) ** 252 - 1
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        risk_data.append({
            'Symbol': symbol,
            'VaR 95%': var_95 * 100,
            'VaR 99%': var_99 * 100,
            'CVaR 95%': cvar_95 * 100,
            'CVaR 99%': cvar_99 * 100,
            'Max Drawdown': max_drawdown * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar
        })
    
    
    risk_df = pd.DataFrame(risk_data)
    
    
    # Interpretation
    best_sharpe = max(risk_data, key=lambda x: x['Sharpe Ratio'])
    worst_drawdown = min(risk_data, key=lambda x: x['Max Drawdown'])
    
    
    return risk_df


def sector_analysis(df):
    """Compare performance and risk across sectors."""
    
    sector_stats = []
    
    for sector, symbols in SECTORS.items():
        all_returns = []
        for symbol in symbols:
            data = df[df['symbol'] == symbol]
            returns = data['change_in_prcntg_try'].dropna().values
            all_returns.extend(returns)
        
        all_returns = np.array(all_returns)
        
        sector_stats.append({
            'Sector': sector,
            'Avg Daily Return (%)': np.mean(all_returns) * 100,
            'Volatility (%)': np.std(all_returns, ddof=1) * 100,
            'Skewness': stats.skew(all_returns),
            'Kurtosis': stats.kurtosis(all_returns),
            'VaR 95% (%)': np.percentile(all_returns, 5) * 100,
            'Max Return (%)': np.max(all_returns) * 100,
            'Min Return (%)': np.min(all_returns) * 100
        })
    
    sector_df = pd.DataFrame(sector_stats)
    
    # Intra-sector correlation
    
    for sector, symbols in SECTORS.items():
        if len(symbols) >= 2:
            data1 = df[df['symbol'] == symbols[0]].sort_values('date')
            data2 = df[df['symbol'] == symbols[1]].sort_values('date')
            
            merged = pd.merge(data1[['date', 'change_in_prcntg_try']], 
                            data2[['date', 'change_in_prcntg_try']], 
                            on='date', suffixes=('_1', '_2'))
            
            corr = np.corrcoef(merged['change_in_prcntg_try_1'].dropna(), 
                              merged['change_in_prcntg_try_2'].dropna())[0, 1]
            
    
    # ANOVA test across sectors
    
    sector_returns = {}
    for sector, symbols in SECTORS.items():
        returns = []
        for symbol in symbols:
            data = df[df['symbol'] == symbol]
            returns.extend(data['change_in_prcntg_try'].dropna().values)
        sector_returns[sector] = np.array(returns)
    
    # Kruskal-Wallis test (non-parametric ANOVA)
    stat, p_value = kruskal(*sector_returns.values())
    
    # Levene test for variance homogeneity
    stat_lev, p_lev = levene(*sector_returns.values())
    
    return sector_df


def currency_impact_analysis(df):
    """Analyze the impact of USD/TRY on stock returns."""
    
    usd_data = df[df['symbol'] == CURRENCY].sort_values('date')
    usd_returns = usd_data.set_index('date')['change_in_prcntg_try']
    
    impact_data = []
    
    for symbol in STOCKS:
        stock_data = df[df['symbol'] == symbol].sort_values('date')
        stock_returns = stock_data.set_index('date')['change_in_prcntg_try']
        stock_prices_try = stock_data.set_index('date')['price']
        stock_prices_usd = stock_data.set_index('date')['price_usd']
        
        # Align data
        common_idx = stock_returns.index.intersection(usd_returns.index)
        sr = stock_returns.loc[common_idx].values
        ur = usd_returns.loc[common_idx].values
        
        # Remove NaN
        mask = ~(np.isnan(sr) | np.isnan(ur))
        sr, ur = sr[mask], ur[mask]
        
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(sr, ur)
        
        # Spearman correlation (rank-based)
        spearman_corr, spearman_p = spearmanr(sr, ur)
        
        # Kendall tau
        kendall_corr, kendall_p = kendalltau(sr, ur)
        
        # Regression: Stock return = α + β * USD return + ε
        slope, intercept, r_value, p_value, std_err = stats.linregress(ur, sr)
        
        # Currency sensitivity (beta to USD)
        beta_usd = np.cov(sr, ur)[0, 1] / np.var(ur)
        
        impact_data.append({
            'Symbol': symbol,
            'Pearson ρ': pearson_corr,
            'Pearson p': pearson_p,
            'Spearman ρ': spearman_corr,
            'Kendall τ': kendall_corr,
            'β to USD': beta_usd,
            'R²': r_value ** 2
        })
    
    
    impact_df = pd.DataFrame(impact_data)
    
    
    # Interpretation
    highest_beta = max(impact_data, key=lambda x: x['β to USD'])
    lowest_beta = min(impact_data, key=lambda x: x['β to USD'])
    
    return impact_df


def correlation_stability(df):
    """Analyze how correlations change over time."""
    
    window = 60  # 60-day rolling window
    
    
    # Prepare data for rolling correlation
    pivot_df = df.pivot(index='date', columns='symbol', values='change_in_prcntg_try')
    
    stability_data = []
    
    for symbol in STOCKS:
        if symbol in pivot_df.columns and CURRENCY in pivot_df.columns:
            rolling_corr = pivot_df[symbol].rolling(window).corr(pivot_df[CURRENCY])
            
            stability_data.append({
                'Symbol': symbol,
                'Mean Corr': rolling_corr.mean(),
                'Std Corr': rolling_corr.std(),
                'Min Corr': rolling_corr.min(),
                'Max Corr': rolling_corr.max(),
                'Range': rolling_corr.max() - rolling_corr.min(),
                'CV (%)': abs(rolling_corr.std() / rolling_corr.mean()) * 100 if rolling_corr.mean() != 0 else 0
            })
    
    stability_df = pd.DataFrame(stability_data)
    
    most_stable = min(stability_data, key=lambda x: x['Std Corr'])
    least_stable = max(stability_data, key=lambda x: x['Std Corr'])
    
    return stability_df


def autocorrelation_analysis(df):
    """Analyze autocorrelation in returns (predictability)."""
    
    
    acf_data = []
    
    for symbol in STOCKS + [CURRENCY]:
        data = df[df['symbol'] == symbol].sort_values('date')
        returns = data['change_in_prcntg_try'].dropna().values
        
        # Calculate autocorrelations for lags 1-5
        acf_values = {}
        for lag in range(1, 6):
            if len(returns) > lag:
                acf = np.corrcoef(returns[lag:], returns[:-lag])[0, 1]
                acf_values[f'ACF({lag})'] = acf
        
        # Ljung-Box test for randomness (lag 10)
        if len(returns) > 10:
            # Manual Ljung-Box Q statistic
            n = len(returns)
            q_stat = 0
            for k in range(1, 11):
                rk = np.corrcoef(returns[k:], returns[:-k])[0, 1]
                q_stat += (rk ** 2) / (n - k)
            q_stat *= n * (n + 2)
            p_value = 1 - stats.chi2.cdf(q_stat, df=10)
        else:
            q_stat, p_value = 0, 1
        
        acf_data.append({
            'Symbol': symbol,
            **acf_values,
            'Q-stat': q_stat,
            'p-value': p_value
        })
    
    acf_df = pd.DataFrame(acf_data)
    
    for row in acf_data:
        sig = "Predictable (reject random walk)" if row['p-value'] < 0.05 else "Random walk (efficient)"
    
    return acf_df


def distribution_tests(df):
    """Advanced distribution analysis and normality tests."""
    
    dist_data = []
    
    for symbol in STOCKS + [CURRENCY]:
        data = df[df['symbol'] == symbol]
        returns = data['change_in_prcntg_try'].dropna().values
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(returns[:5000])  # Shapiro max 5000
        jb_stat, jb_p = jarque_bera(returns)
        anderson_result = anderson(returns, dist='norm')
        
        # D'Agostino-Pearson test
        dagostino_stat, dagostino_p = stats.normaltest(returns)
        
        # Fat tails check
        excess_kurtosis = stats.kurtosis(returns)
        
        dist_data.append({
            'Symbol': symbol,
            'Shapiro-W': shapiro_stat,
            'Shapiro p': shapiro_p,
            'JB Stat': jb_stat,
            'JB p': jb_p,
            "D'Agostino p": dagostino_p,
            'Excess Kurt': excess_kurtosis,
            'Normal?': 'Yes' if shapiro_p > 0.05 else 'No'
        })
    
    
    dist_df = pd.DataFrame(dist_data)
    
    
    
    return dist_df


def cross_correlation_lags(df):
    """Analyze lead-lag relationships between stocks."""
    
    
    pivot_df = df.pivot(index='date', columns='symbol', values='change_in_prcntg_try')
    
    lead_lag_data = []
    
    # Check if GARAN leads/lags AKBNK (same sector)
    pairs = [
        ('GARAN.IS', 'AKBNK.IS', 'Banking'),
        ('ENJSA.IS', 'ZOREN.IS', 'Energy'),
        ('BIMAS.IS', 'ULKER.IS', 'Food'),
        ('GARAN.IS', CURRENCY, 'Bank-USD'),
        ('BIMAS.IS', CURRENCY, 'Food-USD'),
    ]
    
    for stock1, stock2, label in pairs:
        if stock1 in pivot_df.columns and stock2 in pivot_df.columns:
            s1 = pivot_df[stock1].dropna()
            s2 = pivot_df[stock2].dropna()
            
            common = s1.index.intersection(s2.index)
            s1, s2 = s1.loc[common], s2.loc[common]
            
            # Lag -1: stock1 today vs stock2 yesterday (stock1 leads)
            # Lag 0: same day
            # Lag +1: stock1 today vs stock2 tomorrow (stock1 lags)
            
            corr_lag_neg1 = np.corrcoef(s1.values[1:], s2.values[:-1])[0, 1]  # stock1 leads
            corr_lag_0 = np.corrcoef(s1.values, s2.values)[0, 1]  # same day
            corr_lag_pos1 = np.corrcoef(s1.values[:-1], s2.values[1:])[0, 1]  # stock1 lags
            
            lead_lag_data.append({
                'Pair': label,
                'Stock A': stock1,
                'Stock B': stock2,
                'A leads B': corr_lag_neg1,
                'Same Day': corr_lag_0,
                'A lags B': corr_lag_pos1
            })
    
    
    ll_df = pd.DataFrame(lead_lag_data)
    return ll_df


def hypothesis_tests(df):
    """Perform hypothesis tests on stock returns."""
    
    
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol]
        returns = data['change_in_prcntg_try'].dropna().values
        
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        mean_ret = np.mean(returns) * 100
        
        sig = "***" if p_value < 0.01 else ("**" if p_value < 0.05 else ("*" if p_value < 0.1 else ""))
    
    
    banking_returns = []
    food_returns = []
    for symbol in SECTORS['Banking']:
        data = df[df['symbol'] == symbol]
        banking_returns.extend(data['change_in_prcntg_try'].dropna().values)
    for symbol in SECTORS['Food']:
        data = df[df['symbol'] == symbol]
        food_returns.extend(data['change_in_prcntg_try'].dropna().values)
    
    u_stat, u_p = mannwhitneyu(banking_returns, food_returns, alternative='two-sided')
    
    
    sector_returns = []
    for sector in SECTORS.values():
        rets = []
        for symbol in sector:
            data = df[df['symbol'] == symbol]
            rets.extend(data['change_in_prcntg_try'].dropna().values)
        sector_returns.append(rets)
    
    bart_stat, bart_p = bartlett(*sector_returns)


def beta_analysis(df):
    """Calculate market beta for each stock."""
    
    # Use equal-weighted portfolio of all stocks as market proxy
    pivot_df = df.pivot(index='date', columns='symbol', values='change_in_prcntg_try')
    market_return = pivot_df[STOCKS].mean(axis=1)
    
    beta_data = []
    
    for symbol in STOCKS:
        stock_return = pivot_df[symbol]
        
        # Align and clean
        common = stock_return.dropna().index.intersection(market_return.dropna().index)
        sr = stock_return.loc[common].values
        mr = market_return.loc[common].values
        
        # Beta = Cov(stock, market) / Var(market)
        cov = np.cov(sr, mr)[0, 1]
        var_market = np.var(mr, ddof=1)
        beta = cov / var_market
        
        # Alpha (Jensen's alpha)
        alpha = np.mean(sr) - beta * np.mean(mr)
        
        # R-squared
        r_squared = np.corrcoef(sr, mr)[0, 1] ** 2
        
        # Treynor ratio
        risk_free_daily = ((1 + 0.40) ** (1/252)) - 1
        treynor = (np.mean(sr) - risk_free_daily) / beta if beta != 0 else 0
        
        beta_data.append({
            'Symbol': symbol,
            'Beta': beta,
            'Alpha (daily)': alpha * 100,
            'R²': r_squared,
            'Treynor': treynor * 252  # Annualized
        })
    
    
    beta_df = pd.DataFrame(beta_data)
    
    aggressive = [b for b in beta_data if b['Beta'] > 1]
    defensive = [b for b in beta_data if b['Beta'] < 1]
    
    return beta_df

def create_separate_plots(df):
    """Create separate visualization plots and save to img/task_8/."""
    plot_dir = "img/task_8"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    pivot_df = df.pivot(index='date', columns='symbol', values='change_in_prcntg_try')
    
    # PLOT 1: Return Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for symbol in STOCKS[:3]:
        data = df[df['symbol'] == symbol]
        returns = data['change_in_prcntg_try'].dropna() * 100
        axes[0].hist(returns, bins=50, alpha=0.5, label=symbol.replace('.IS', ''), density=True)
    axes[0].set_xlabel('Daily Return (%)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Return Distributions: GARAN, AKBNK, ENJSA', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(-15, 15)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    for symbol in STOCKS[3:]:
        data = df[df['symbol'] == symbol]
        returns = data['change_in_prcntg_try'].dropna() * 100
        axes[1].hist(returns, bins=50, alpha=0.5, label=symbol.replace('.IS', ''), density=True)
    axes[1].set_xlabel('Daily Return (%)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Return Distributions: ZOREN, BIMAS, ULKER', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(-15, 15)
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_01_return_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_01_return_distributions.png")
    
    # PLOT 2: Rolling Volatility
    fig, ax = plt.subplots(figsize=(14, 6))
    for symbol in STOCKS:
        rolling_vol = pivot_df[symbol].rolling(30).std() * np.sqrt(252) * 100
        ax.plot(pivot_df.index, rolling_vol, label=symbol.replace('.IS', ''), linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
    ax.set_title('30-Day Rolling Volatility (Annualized)', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_02_rolling_volatility.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_02_rolling_volatility.png")
    
    # PLOT 3: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = pivot_df[STOCKS + [CURRENCY]].corr()
    labels = [s.replace('.IS', '').replace('=X', '') for s in STOCKS + [CURRENCY]]
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax, annot_kws={'size': 11}, xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix: Daily Returns', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_03_correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_03_correlation_heatmap.png")
    
    # PLOT 4: Risk-Return Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(STOCKS)))
    for i, symbol in enumerate(STOCKS):
        data = df[df['symbol'] == symbol]
        returns = data['change_in_prcntg_try'].dropna()
        mean_ret = np.mean(returns) * 252 * 100
        vol = np.std(returns, ddof=1) * np.sqrt(252) * 100
        ax.scatter(vol, mean_ret, s=200, c=[colors[i]], label=symbol.replace('.IS', ''), edgecolors='black')
        ax.annotate(symbol.replace('.IS', ''), (vol, mean_ret), fontsize=10, 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
    ax.set_ylabel('Annualized Return (%)', fontsize=12)
    ax.set_title('Risk-Return Profile: All Stocks', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_04_risk_return.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_04_risk_return.png")
    
    # PLOT 5: Cumulative Returns
    fig, ax = plt.subplots(figsize=(14, 6))
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol].sort_values('date')
        returns = data['change_in_prcntg_try'].fillna(0).values
        cum_ret = (np.cumprod(1 + returns) - 1) * 100
        ax.plot(data['date'], cum_ret, label=symbol.replace('.IS', ''), linewidth=1.5, alpha=0.9)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax.set_title('Cumulative Returns Over Time (TRY)', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_05_cumulative_returns.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_05_cumulative_returns.png")
    
    # PLOT 6: Rolling Correlation with USD
    fig, ax = plt.subplots(figsize=(14, 6))
    for symbol in STOCKS:
        rolling_corr = pivot_df[symbol].rolling(60).corr(pivot_df[CURRENCY])
        ax.plot(pivot_df.index, rolling_corr, label=symbol.replace('.IS', ''), linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Correlation Coefficient', fontsize=11)
    ax.set_title('60-Day Rolling Correlation with USD/TRY', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_06_rolling_usd_correlation.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_06_rolling_usd_correlation.png")
    
    # PLOT 7: Boxplot Returns
    fig, ax = plt.subplots(figsize=(12, 6))
    sector_data = []
    sector_labels = []
    sector_colors = []
    color_map = {'Banking': 'steelblue', 'Energy': 'darkorange', 'Food': 'forestgreen'}
    for sector, symbols in SECTORS.items():
        for symbol in symbols:
            data = df[df['symbol'] == symbol]
            returns = data['change_in_prcntg_try'].dropna() * 100
            sector_data.append(returns.values)
            sector_labels.append(f"{symbol.replace('.IS', '')}\n({sector})")
            sector_colors.append(color_map[sector])
    
    bp = ax.boxplot(sector_data, tick_labels=sector_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], sector_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Daily Return (%)', fontsize=11)
    ax.set_title('Daily Return Distribution by Stock', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_07_boxplot_returns.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_07_boxplot_returns.png")
    
    # PLOT 8: VaR Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    var_data = []
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol]
        returns = data['change_in_prcntg_try'].dropna().values
        var_95 = np.percentile(returns, 5) * 100
        var_data.append((symbol.replace('.IS', ''), var_95))
    var_df = pd.DataFrame(var_data, columns=['Stock', 'VaR 95%'])
    var_df = var_df.sort_values('VaR 95%')
    colors = ['crimson' if v < var_df['VaR 95%'].median() else 'coral' for v in var_df['VaR 95%']]
    ax.barh(var_df['Stock'], var_df['VaR 95%'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('VaR 95% (Daily Loss %)', fontsize=11)
    ax.set_title('Value at Risk (95% Confidence)', fontsize=14)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for i, (stock, var) in enumerate(zip(var_df['Stock'], var_df['VaR 95%'])):
        ax.text(var - 0.1, i, f'{var:.2f}%', va='center', ha='right', fontsize=10, color='white', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_08_var_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_08_var_comparison.png")
    
    # PLOT 9: Drawdown Analysis
    fig, ax = plt.subplots(figsize=(14, 6))
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol].sort_values('date')
        returns = data['change_in_prcntg_try'].fillna(0).values
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max * 100
        ax.fill_between(data['date'], drawdowns, 0, alpha=0.3, label=symbol.replace('.IS', ''))
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title('Drawdown Analysis: All Stocks', fontsize=14)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_09_drawdown.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_09_drawdown.png")
    
    # PLOT 10: Autocorrelation Function
    fig, ax = plt.subplots(figsize=(10, 6))
    lags = range(1, 11)
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol]
        returns = data['change_in_prcntg_try'].dropna().values
        acf_vals = [np.corrcoef(returns[lag:], returns[:-lag])[0, 1] for lag in lags]
        ax.plot(lags, acf_vals, 'o-', label=symbol.replace('.IS', ''), alpha=0.7, markersize=6)
    
    n = len(df[df['symbol'] == STOCKS[0]]['change_in_prcntg_try'].dropna())
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(y=1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.7, label='95% CI')
    ax.axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Lag (Days)', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.set_title('Autocorrelation Function (ACF)', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(lags)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_10_autocorrelation.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_10_autocorrelation.png")
    
    # PLOT 11: Market Beta
    fig, ax = plt.subplots(figsize=(10, 6))
    market_return = pivot_df[STOCKS].mean(axis=1)
    betas = []
    for symbol in STOCKS:
        stock_return = pivot_df[symbol]
        common = stock_return.dropna().index.intersection(market_return.dropna().index)
        sr = stock_return.loc[common].values
        mr = market_return.loc[common].values
        beta = np.cov(sr, mr)[0, 1] / np.var(mr, ddof=1)
        betas.append((symbol.replace('.IS', ''), beta))
    beta_df = pd.DataFrame(betas, columns=['Stock', 'Beta'])
    beta_df = beta_df.sort_values('Beta')
    colors = ['forestgreen' if b < 1 else 'crimson' for b in beta_df['Beta']]
    ax.barh(beta_df['Stock'], beta_df['Beta'], color=colors, alpha=0.8, edgecolor='black')
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Market (β=1)')
    ax.set_xlabel('Beta', fontsize=11)
    ax.set_title('Market Beta: Sensitivity to Market Portfolio', fontsize=14)
    ax.legend(fontsize=10)
    for i, (stock, beta) in enumerate(zip(beta_df['Stock'], beta_df['Beta'])):
        ax.text(beta + 0.02, i, f'{beta:.2f}', va='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_11_beta.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_11_beta.png")
    
    # PLOT 12: Sector Comparison (Violin Plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    sector_returns_list = []
    sector_names = []
    for sector, symbols in SECTORS.items():
        returns = []
        for symbol in symbols:
            data = df[df['symbol'] == symbol]
            returns.extend(data['change_in_prcntg_try'].dropna().values * 100)
        sector_returns_list.append(returns)
        sector_names.append(sector)
    
    parts = ax.violinplot(sector_returns_list, positions=range(len(sector_names)), showmeans=True, showmedians=True)
    colors = ['steelblue', 'darkorange', 'forestgreen']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(sector_names)))
    ax.set_xticklabels(sector_names, fontsize=11)
    ax.set_ylabel('Daily Return (%)', fontsize=11)
    ax.set_title('Return Distribution by Sector (Violin Plot)', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_8_12_sector_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {plot_dir}/task_8_12_sector_comparison.png")
    
    print(f"\nAll 12 plots saved to {plot_dir}/")


def create_visualizations(df):
    """Create comprehensive visualization plots (combined version)."""
    if not os.path.exists("img"):
        os.makedirs("img")
    
    # Now just call the separate plots function
    create_separate_plots(df)


def main():
    """Main execution function."""
    print("="*60)
    print("TASK 8: COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Output buffer
    output = []
    
    # Run all analyses
    print("\n[1/11] Calculating descriptive statistics...")
    descriptive_statistics(df)
    
    print("[2/11] Calculating risk metrics...")
    risk_metrics(df)
    
    print("[3/11] Performing sector analysis...")
    sector_analysis(df)
    
    print("[4/11] Analyzing currency impact...")
    currency_impact_analysis(df)
    
    print("[5/11] Analyzing correlation stability...")
    correlation_stability(df)
    
    print("[6/11] Performing autocorrelation analysis...")
    autocorrelation_analysis(df)
    
    print("[7/11] Running distribution tests...")
    distribution_tests(df)
    
    print("[8/11] Analyzing lead-lag relationships...")
    cross_correlation_lags(df)
    
    print("[9/11] Running hypothesis tests...")
    hypothesis_tests(df)
    
    print("[10/11] Calculating beta analysis...")
    beta_analysis(df)
    
    print("[11/11] Writing summary and discussion...")
    summary_and_discussion(df)
    
    # Add plot explanations to output
    print("[12/12] Adding plot explanations...")
    plot_explanations(output)
    
    # Save to file
    with open("task_8_results.txt", "w") as f:
        f.write("\n".join(output))
    print("\nResults saved to task_8_results.txt")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df)
    
    print("\n" + "="*60)
    print("TASK 8 COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
