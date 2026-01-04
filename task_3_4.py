import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import CONSTS
from helper import helper


def load_data() -> pd.DataFrame:
    """Load the main data.csv file."""
    return pd.read_csv("data.csv", parse_dates=['date'])


def get_change_series(df: pd.DataFrame, symbol: str, change_col: str) -> pd.Series:
    """Extract change series for a given symbol from the dataframe."""
    symbol_df = df[df['symbol'] == symbol].set_index('date').sort_index()
    return symbol_df[change_col].dropna()


def plot_correlation_matrix(change_df: pd.DataFrame, title: str, filename: str) -> None:
    """Plot correlation matrix for the given change data."""
    if not os.path.exists("img"):
        os.makedirs("img")

    symbols = change_df.columns
    n = len(symbols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=symbols, columns=symbols)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            else:
                # Using the custom helper function
                val = helper.calculate_correlation(
                    change_df.iloc[:, i].values, 
                    change_df.iloc[:, j].values
                )
                corr_matrix.iloc[i, j] = val
                if i == 0 and j == 1:
                    print(f"Sample Correlation ({symbols[i]} vs {symbols[j]}): {val:.4f}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"img/{filename}")
    plt.close()
    print(f"Correlation matrix saved to img/{filename}")
    print("\nCorrelation Matrix Values:")
    print(corr_matrix)


def main():
    # Load data from data.csv
    df = load_data()
    print(f"Loaded data.csv with {len(df)} rows.")
    
    stock_symbols = list(CONSTS.STOCKS.keys())
    all_symbols = stock_symbols + [CONSTS.CURRENCY]
    
    # TASK 3: Correlation between daily price changes (TRY) and USD/TRY rate changes
    print("\n" + "="*60)
    print("TASK 3: Daily Price Changes (TRY) vs USD/TRY Change Correlation")
    print("="*60)
    
    # Get USD/TRY changes
    usd_changes = get_change_series(df, CONSTS.CURRENCY, 'change_in_prcntg_try')
    
    task3_results = []
    for symbol in stock_symbols:
        stock_changes = get_change_series(df, symbol, 'change_in_prcntg_try')
        # Align indices
        aligned = pd.concat([stock_changes, usd_changes], axis=1).dropna()
        corr = helper.calculate_correlation(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
        task3_results.append({'Stock': symbol, 'Sector': CONSTS.STOCKS[symbol], 'Correlation with USD/TRY Change': corr})
        print(f"  {symbol} ({CONSTS.STOCKS[symbol]}): {corr:.4f}")
    
    # Build full changes dataframe for matrix
    data_changes = {}
    for symbol in all_symbols:
        data_changes[symbol] = get_change_series(df, symbol, 'change_in_prcntg_try')
    
    df_changes = pd.DataFrame(data_changes).dropna()
    print("\n--- Daily Changes (TRY) Head ---")
    print(df_changes.head())
    plot_correlation_matrix(df_changes, "Correlation of Daily Changes (TRY)", "task_3_correlation_changes_try.png")
    
    # TASK 4: Correlation between daily price changes in USD
    print("\n" + "="*60)
    print("TASK 4: Daily Price Changes (USD) Correlation Matrix")
    print("="*60)
    
    data_usd_changes = {}
    for symbol in stock_symbols:
        data_usd_changes[symbol] = get_change_series(df, symbol, 'change_in_prcntg_usd')
    
    df_usd_changes = pd.DataFrame(data_usd_changes).dropna()
    
    # Print pairwise correlations
    n = len(stock_symbols)
    task4_results = []
    for i in range(n):
        for j in range(i+1, n):
            corr = helper.calculate_correlation(df_usd_changes.iloc[:, i].values, df_usd_changes.iloc[:, j].values)
            task4_results.append({'Stock 1': stock_symbols[i], 'Stock 2': stock_symbols[j], 'Correlation': corr})
            print(f"  {stock_symbols[i]} vs {stock_symbols[j]}: {corr:.4f}")
    
    print("\n--- Daily Changes (USD) Head ---")
    print(df_usd_changes.head())
    plot_correlation_matrix(df_usd_changes, "Correlation of Daily Changes (USD)", "task_4_correlation_changes_usd.png")

if __name__ == "__main__":
    main()
