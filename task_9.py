"""
Sector Analysis, Volatility, and Risk-Reward Analysis
==============================================================
Focused analysis on:
1. Sector-level performance comparison
2. Individual stock volatility with detailed numbers
3. Risk-reward analysis for investment decisions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
SECTOR_COLORS = {'Banking': '#1f77b4', 'Energy': '#ff7f0e', 'Food': '#2ca02c'}
CURRENCY = 'USDTRY=X'


def load_data():
    """Load and prepare data from CSV."""
    df = pd.read_csv('data.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded data.csv with {len(df)} rows.")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def get_stock_sector(symbol):
    """Get sector for a given stock symbol."""
    for sector, symbols in SECTORS.items():
        if symbol in symbols:
            return sector
    return 'Unknown'


# =============================================================================
# ANALYSIS 1: SECTOR ANALYSIS
# =============================================================================
def sector_analysis(df):
    """Comprehensive sector-level analysis."""


    # Calculate sector-level statistics
    sector_stats = []
    sector_returns_data = {}
    
    for sector, symbols in SECTORS.items():
        all_returns = []
        all_prices = []
        
        for symbol in symbols:
            data = df[df['symbol'] == symbol]
            returns = data['change_in_prcntg_try'].dropna().values
            prices = data['price'].values
            all_returns.extend(returns)
            all_prices.extend(prices)
        
        all_returns = np.array(all_returns)
        sector_returns_data[sector] = all_returns
        
        # Calculate statistics
        daily_return = np.mean(all_returns)
        annual_return = ((1 + daily_return) ** 252 - 1) * 100
        daily_vol = np.std(all_returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(252) * 100
        sharpe = (annual_return / 100) / (annual_vol / 100) if annual_vol > 0 else 0
        
        sector_stats.append({
            'Sector': sector,
            'Stocks': ', '.join([s.replace('.IS', '') for s in symbols]),
            'Avg Daily Return (%)': daily_return * 100,
            'Annual Return (%)': annual_return,
            'Daily Volatility (%)': daily_vol * 100,
            'Annual Volatility (%)': annual_vol,
            'Sharpe Ratio': sharpe,
            'Skewness': stats.skew(all_returns),
            'Kurtosis': stats.kurtosis(all_returns),
            'VaR 95% (%)': np.percentile(all_returns, 5) * 100,
            'Best Day (%)': np.max(all_returns) * 100,
            'Worst Day (%)': np.min(all_returns) * 100,
            'N Observations': len(all_returns)
        })
    
    
    sector_df = pd.DataFrame(sector_stats)
    
    # Performance metrics
    
    
    
    
    # Intra-sector correlation
    
    for sector, symbols in SECTORS.items():
        data1 = df[df['symbol'] == symbols[0]].sort_values('date')
        data2 = df[df['symbol'] == symbols[1]].sort_values('date')
        
        merged = pd.merge(data1[['date', 'change_in_prcntg_try']], 
                        data2[['date', 'change_in_prcntg_try']], 
                        on='date', suffixes=('_1', '_2'))
        
        r1 = merged['change_in_prcntg_try_1'].dropna().values
        r2 = merged['change_in_prcntg_try_2'].dropna().values
        
        if len(r1) > 0 and len(r2) > 0:
            corr = np.corrcoef(r1, r2)[0, 1]
    
    # Cross-sector correlation
    
    cross_corr = []
    sectors_list = list(SECTORS.keys())
    for i, sector1 in enumerate(sectors_list):
        for sector2 in sectors_list[i+1:]:
            r1 = sector_returns_data[sector1]
            r2 = sector_returns_data[sector2]
            
            # Use minimum length
            min_len = min(len(r1), len(r2))
            corr = np.corrcoef(r1[:min_len], r2[:min_len])[0, 1]
            cross_corr.append({
                'Sector 1': sector1,
                'Sector 2': sector2,
                'Correlation': corr
            })
    
    return sector_df, sector_returns_data


# =============================================================================
# ANALYSIS 2: VOLATILITY ANALYSIS
# =============================================================================
def volatility_analysis(df):
    """Detailed volatility analysis for each stock."""

    
    vol_data = []
    
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol].sort_values('date')
        returns = data['change_in_prcntg_try'].dropna().values
        prices = data['price'].values
        
        # Calculate various volatility measures
        daily_vol = np.std(returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Downside volatility (only negative returns)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0
        
        # Upside volatility (only positive returns)
        upside_returns = returns[returns > 0]
        upside_vol = np.std(upside_returns, ddof=1) if len(upside_returns) > 0 else 0
        
        # Rolling volatility stats
        rolling_vol = pd.Series(returns).rolling(30).std()
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        vol_data.append({
            'Symbol': symbol,
            'Sector': get_stock_sector(symbol),
            'Daily Vol (%)': daily_vol * 100,
            'Annual Vol (%)': annual_vol * 100,
            'Downside Vol (%)': downside_vol * 100,
            'Upside Vol (%)': upside_vol * 100,
            'Vol Ratio (Up/Down)': upside_vol / downside_vol if downside_vol > 0 else 0,
            'Avg Rolling Vol (%)': rolling_vol.mean() * 100 if not rolling_vol.isna().all() else 0,
            'Max Rolling Vol (%)': rolling_vol.max() * 100 if not rolling_vol.isna().all() else 0,
            'Min Rolling Vol (%)': rolling_vol.min() * 100 if not rolling_vol.isna().all() else 0,
            'Max Drawdown (%)': max_drawdown * 100,
            'VaR 95% (%)': np.percentile(returns, 5) * 100,
            'VaR 99% (%)': np.percentile(returns, 1) * 100,
            'N Days': len(returns)
        })
    
    vol_df = pd.DataFrame(vol_data)

    return vol_df


# =============================================================================
# ANALYSIS 3: RISK-REWARD ANALYSIS
# =============================================================================
def risk_reward_analysis(df):
    """Risk-reward analysis for each stock."""
    risk_free_annual = 0.40  # 40% annual risk-free rate
    risk_free_daily = (1 + risk_free_annual) ** (1/252) - 1
    
    rr_data = []
    
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol].sort_values('date')
        returns = data['change_in_prcntg_try'].dropna().values
        
        # Basic return stats
        mean_daily = np.mean(returns)
        annual_return = (1 + mean_daily) ** 252 - 1
        
        # Volatility
        daily_vol = np.std(returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Downside volatility
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0.0001
        annual_downside_vol = downside_vol * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Risk-adjusted ratios
        excess_return = annual_return - risk_free_annual
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0
        sortino = excess_return / annual_downside_vol if annual_downside_vol > 0 else 0
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win/Loss analysis
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        
        win_rate = len(winning_days) / len(returns) * 100 if len(returns) > 0 else 0
        avg_win = np.mean(winning_days) * 100 if len(winning_days) > 0 else 0
        avg_loss = abs(np.mean(losing_days)) * 100 if len(losing_days) > 0 else 0.0001
        
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Expectancy (expected value per trade)
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        
        rr_data.append({
            'Symbol': symbol,
            'Sector': get_stock_sector(symbol),
            'Annual Return (%)': annual_return * 100,
            'Annual Vol (%)': annual_vol * 100,
            'Excess Return (%)': excess_return * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Max Drawdown (%)': max_drawdown * 100,
            'Win Rate (%)': win_rate,
            'Avg Win (%)': avg_win,
            'Avg Loss (%)': avg_loss,
            'Risk-Reward Ratio': risk_reward_ratio,
            'Expectancy (%)': expectancy
        })
    
    rr_df = pd.DataFrame(rr_data)
    
    
    
    
    # Detailed interpretation
    
    for r in rr_data:
        
        # Sharpe interpretation
        if r['Sharpe Ratio'] > 1:
            sharpe_grade = "EXCELLENT"
        elif r['Sharpe Ratio'] > 0.5:
            sharpe_grade = "GOOD"
        elif r['Sharpe Ratio'] > 0:
            sharpe_grade = "FAIR"
        else:
            sharpe_grade = "POOR"
        
        
        
        if r['Risk-Reward Ratio'] > 1:
            rr_assessment = "FAVORABLE"
        else:
            rr_assessment = "UNFAVORABLE"
        
        if r['Expectancy (%)'] > 0:
            expectancy_assessment = "POSITIVE"
        else:
            expectancy_assessment = "NEGATIVE"
    
    return rr_df


# =============================================================================
# VISUALIZATIONS
# =============================================================================
def create_plots(df, sector_df, vol_df, rr_df):
    """Create three visualization plots."""
    plot_dir = "img/task_9"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    pivot_df = df.pivot(index='date', columns='symbol', values='change_in_prcntg_try')
    
    # =========================================================================
    # PLOT 1: SECTOR ANALYSIS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sector Analysis', fontsize=16, fontweight='bold')
    
    # 1a: Sector Returns Comparison (Bar Chart)
    ax1 = axes[0, 0]
    sectors = sector_df['Sector'].values
    annual_returns = sector_df['Annual Return (%)'].values
    colors = [SECTOR_COLORS[s] for s in sectors]
    bars = ax1.bar(sectors, annual_returns, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Annual Return (%)', fontsize=11)
    ax1.set_title('Sector Annual Returns', fontsize=12)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, annual_returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 1b: Sector Volatility Comparison
    ax2 = axes[0, 1]
    annual_vol = sector_df['Annual Volatility (%)'].values
    bars = ax2.bar(sectors, annual_vol, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Annual Volatility (%)', fontsize=11)
    ax2.set_title('Sector Volatility (Risk)', fontsize=12)
    for bar, val in zip(bars, annual_vol):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 1c: Sector Cumulative Returns Over Time
    ax3 = axes[1, 0]
    for sector, symbols in SECTORS.items():
        # Calculate sector average cumulative return
        sector_returns = []
        for symbol in symbols:
            data = df[df['symbol'] == symbol].sort_values('date')
            returns = data['change_in_prcntg_try'].fillna(0).values
            cum_ret = np.cumprod(1 + returns) - 1
            sector_returns.append(cum_ret)
        
        # Average of the two stocks
        min_len = min(len(r) for r in sector_returns)
        avg_cum = np.mean([r[:min_len] for r in sector_returns], axis=0) * 100
        dates = df[df['symbol'] == symbols[0]].sort_values('date')['date'].values[:min_len]
        
        ax3.plot(dates, avg_cum, label=sector, color=SECTOR_COLORS[sector], linewidth=2)
    
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax3.set_title('Sector Cumulative Returns Over Time', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 1d: Sector Violin Plot
    ax4 = axes[1, 1]
    sector_returns_list = []
    sector_names = []
    for sector, symbols in SECTORS.items():
        returns = []
        for symbol in symbols:
            data = df[df['symbol'] == symbol]
            returns.extend(data['change_in_prcntg_try'].dropna().values * 100)
        sector_returns_list.append(returns)
        sector_names.append(sector)
    
    parts = ax4.violinplot(sector_returns_list, positions=range(len(sector_names)), 
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(SECTOR_COLORS.values())[i])
        pc.set_alpha(0.7)
    ax4.set_xticks(range(len(sector_names)))
    ax4.set_xticklabels(sector_names, fontsize=11)
    ax4.set_ylabel('Daily Return (%)', fontsize=11)
    ax4.set_title('Sector Return Distributions', fontsize=12)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_9_01_sector_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_dir}/task_9_01_sector_analysis.png")
    
    # =========================================================================
    # PLOT 2: VOLATILITY ANALYSIS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Volatility Analysis', fontsize=16, fontweight='bold')
    
    # 2a: Annual Volatility by Stock
    ax1 = axes[0, 0]
    vol_sorted = vol_df.sort_values('Annual Vol (%)')
    colors = [SECTOR_COLORS[s] for s in vol_sorted['Sector']]
    bars = ax1.barh(vol_sorted['Symbol'].str.replace('.IS', ''), 
                    vol_sorted['Annual Vol (%)'], color=colors, edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Annual Volatility (%)', fontsize=11)
    ax1.set_title('Stock Volatility Comparison', fontsize=12)
    for i, (bar, val) in enumerate(zip(bars, vol_sorted['Annual Vol (%)'])):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add legend for sectors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=SECTOR_COLORS[s], edgecolor='black', label=s) 
                      for s in SECTORS.keys()]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # 2b: Rolling Volatility Over Time
    ax2 = axes[0, 1]
    for symbol in STOCKS:
        data = df[df['symbol'] == symbol].sort_values('date')
        returns = data['change_in_prcntg_try'].values
        dates = data['date'].values
        rolling_vol = pd.Series(returns).rolling(30).std() * np.sqrt(252) * 100
        sector = get_stock_sector(symbol)
        ax2.plot(dates, rolling_vol, label=symbol.replace('.IS', ''), 
                color=SECTOR_COLORS[sector], alpha=0.7, linewidth=1.2)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('30-Day Rolling Volatility (%)', fontsize=11)
    ax2.set_title('Volatility Over Time', fontsize=12)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 2c: VaR Comparison
    ax3 = axes[1, 0]
    var_sorted = vol_df.sort_values('VaR 95% (%)')
    colors = [SECTOR_COLORS[s] for s in var_sorted['Sector']]
    bars = ax3.barh(var_sorted['Symbol'].str.replace('.IS', ''), 
                    var_sorted['VaR 95% (%)'], color=colors, edgecolor='black', alpha=0.8)
    ax3.set_xlabel('VaR 95% (Daily Loss %)', fontsize=11)
    ax3.set_title('Value at Risk (95% Confidence)', fontsize=12)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, var_sorted['VaR 95% (%)']):
        ax3.text(val - 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}%', va='center', ha='right', fontsize=10, 
                fontweight='bold', color='white')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 2d: Max Drawdown Comparison
    ax4 = axes[1, 1]
    dd_sorted = vol_df.sort_values('Max Drawdown (%)')
    colors = [SECTOR_COLORS[s] for s in dd_sorted['Sector']]
    bars = ax4.barh(dd_sorted['Symbol'].str.replace('.IS', ''), 
                    dd_sorted['Max Drawdown (%)'], color=colors, edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Maximum Drawdown (%)', fontsize=11)
    ax4.set_title('Worst Peak-to-Trough Decline', fontsize=12)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, dd_sorted['Max Drawdown (%)']):
        ax4.text(val - 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', ha='right', fontsize=10, 
                fontweight='bold', color='white')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_9_02_volatility_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_dir}/task_9_02_volatility_analysis.png")
    
    # =========================================================================
    # PLOT 3: RISK-REWARD ANALYSIS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Risk-Reward Analysis', fontsize=16, fontweight='bold')
    
    # 3a: Risk-Return Scatter Plot
    ax1 = axes[0, 0]
    for _, row in rr_df.iterrows():
        sector = row['Sector']
        ax1.scatter(row['Annual Vol (%)'], row['Annual Return (%)'], 
                   s=200, c=SECTOR_COLORS[sector], edgecolors='black', linewidth=1.5, alpha=0.8)
        ax1.annotate(row['Symbol'].replace('.IS', ''), 
                    (row['Annual Vol (%)'], row['Annual Return (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax1.set_xlabel('Annual Volatility (Risk) %', fontsize=11)
    ax1.set_ylabel('Annual Return (Reward) %', fontsize=11)
    ax1.set_title('Risk-Return Trade-off', fontsize=12)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=40, color='red', linestyle=':', alpha=0.5, label='Risk-free rate (40%)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add legend for sectors
    legend_elements = [Patch(facecolor=SECTOR_COLORS[s], edgecolor='black', label=s) 
                      for s in SECTORS.keys()]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # 3b: Sharpe Ratio Comparison
    ax2 = axes[0, 1]
    sharpe_sorted = rr_df.sort_values('Sharpe Ratio')
    colors = [SECTOR_COLORS[s] for s in sharpe_sorted['Sector']]
    bars = ax2.barh(sharpe_sorted['Symbol'].str.replace('.IS', ''), 
                    sharpe_sorted['Sharpe Ratio'], color=colors, edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Sharpe Ratio', fontsize=11)
    ax2.set_title('Risk-Adjusted Return (Sharpe Ratio)', fontsize=12)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, sharpe_sorted['Sharpe Ratio']):
        x_pos = val + 0.02 if val >= 0 else val - 0.02
        ha = 'left' if val >= 0 else 'right'
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', ha=ha, fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3c: Win Rate vs Risk-Reward Ratio
    ax3 = axes[1, 0]
    for _, row in rr_df.iterrows():
        sector = row['Sector']
        ax3.scatter(row['Win Rate (%)'], row['Risk-Reward Ratio'], 
                   s=200, c=SECTOR_COLORS[sector], edgecolors='black', linewidth=1.5, alpha=0.8)
        ax3.annotate(row['Symbol'].replace('.IS', ''), 
                    (row['Win Rate (%)'], row['Risk-Reward Ratio']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax3.set_xlabel('Win Rate (%)', fontsize=11)
    ax3.set_ylabel('Risk-Reward Ratio (Avg Win / Avg Loss)', fontsize=11)
    ax3.set_title('Win Rate vs Risk-Reward', fontsize=12)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even line')
    ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 3d: Expectancy Comparison
    ax4 = axes[1, 1]
    exp_sorted = rr_df.sort_values('Expectancy (%)')
    colors_exp = ['forestgreen' if e > 0 else 'crimson' for e in exp_sorted['Expectancy (%)']]
    bars = ax4.barh(exp_sorted['Symbol'].str.replace('.IS', ''), 
                    exp_sorted['Expectancy (%)'], color=colors_exp, edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Expectancy (Expected Daily Return %)', fontsize=11)
    ax4.set_title('Expected Daily Gain/Loss', fontsize=12)
    ax4.axvline(x=0, color='gray', linestyle='-', linewidth=2)
    for bar, val in zip(bars, exp_sorted['Expectancy (%)']):
        x_pos = val + 0.002 if val >= 0 else val - 0.002
        ha = 'left' if val >= 0 else 'right'
        ax4.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}%', va='center', ha=ha, fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/task_9_03_risk_reward_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_dir}/task_9_03_risk_reward_analysis.png")
    
    print(f"\nAll 3 plots saved to {plot_dir}/")



def main():
    """Main execution function."""
    print("="*60)
    print("SECTOR, VOLATILITY & RISK-REWARD ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Output buffer
    output = []
    
    # Run analyses
    print("\n[1/3] Performing sector analysis...")
    sector_df, sector_returns_data = sector_analysis(df)
    
    print("[2/3] Performing volatility analysis...")
    vol_df = volatility_analysis(df)
    
    print("[3/3] Performing risk-reward analysis...")
    rr_df = risk_reward_analysis(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_plots(df, sector_df, vol_df, rr_df)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
