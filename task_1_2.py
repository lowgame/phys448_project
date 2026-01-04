#!/usr/bin/env python3

# Hocam do not forget to create venv:
# python3 -m venv venv
# source venv/bin/activate
# pip3 install uv 
# "uv" is a package manager like pip but way more faster.
# uv pip install -r requirements.txt

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


def get_price_series(df: pd.DataFrame, symbol: str, price_col: str = 'price') -> pd.Series:
    """Extract price series for a given symbol from the dataframe."""
    symbol_df = df[df['symbol'] == symbol].set_index('date').sort_index()
    return symbol_df[price_col]


def find_correlations(df: pd.DataFrame, label_1: str, label_2: str) -> pd.DataFrame:
    """Calculate correlation statistics between two symbols."""
    data_1 = get_price_series(df, label_1).values
    data_2 = get_price_series(df, label_2).values
    
    correlation = helper.calculate_correlation(data_1, data_2)
    covariance = helper.calculate_covariance(data_1, data_2)
    variance_1 = helper.calculate_variance(data_1)
    variance_2 = helper.calculate_variance(data_2)

    results = {
        "Label 1": label_1,
        "Label 2": label_2,
        "Covariance": covariance,
        "Variance 1": variance_1,
        "Variance 2": variance_2,
        "Correlation": correlation
    }
    return pd.DataFrame([results])


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot correlation matrix for stock prices in TRY (including USD/TRY rate)."""
    if not os.path.exists("img"):
        os.makedirs("img")

    stock_symbols = list(CONSTS.STOCKS.keys())
    all_symbols = stock_symbols + [CONSTS.CURRENCY]
    
    # Build a DataFrame with prices for each symbol
    data = {}
    for symbol in all_symbols:
        data[symbol] = get_price_series(df, symbol, 'price')
    
    price_df = pd.DataFrame(data)
    
    # Create correlation matrix using helper.calculate_correlation
    symbols = price_df.columns
    n = len(symbols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=symbols, columns=symbols)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            else:
                corr_matrix.iloc[i, j] = helper.calculate_correlation(
                    price_df.iloc[:, i].values, 
                    price_df.iloc[:, j].values
                )

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Stock Correlation Matrix (TRY Prices + USD/TRY Rate)")
    plt.tight_layout()
    plt.savefig("img/task_1_correlation_matrix.png")
    plt.close()
    print("Correlation matrix image saved to img/task_1_correlation_matrix.png")


def plot_usd_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot correlation matrix for stock prices in USD."""
    if not os.path.exists("img"):
        os.makedirs("img")

    stock_symbols = list(CONSTS.STOCKS.keys())
    
    # Build a DataFrame with USD prices for each symbol
    data_usd = {}
    for symbol in stock_symbols:
        data_usd[symbol] = get_price_series(df, symbol, 'price_usd')
    
    df_usd = pd.DataFrame(data_usd)
    
    # Create correlation matrix using helper.calculate_correlation
    symbols = df_usd.columns
    n = len(symbols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=symbols, columns=symbols)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            else:
                corr_matrix.iloc[i, j] = helper.calculate_correlation(
                    df_usd.iloc[:, i].values, 
                    df_usd.iloc[:, j].values
                )

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Stock Correlation Matrix (USD Prices)")
    plt.tight_layout()
    plt.savefig("img/task_2_usd_correlation_matrix.png")
    plt.close()
    print("USD Correlation matrix image saved to img/task_2_usd_correlation_matrix.png")


def main():
    # Load data from data.csv
    df = load_data()
    print(f"Loaded data.csv with {len(df)} rows.")
    
    # TASK 1: Correlation between stock prices (TRY) and USD/TRY rate
    print("\n" + "="*60)
    print("TASK 1: Stock Prices (TRY) vs USD/TRY Exchange Rate Correlation")
    print("="*60)
    
    stock_symbols = list(CONSTS.STOCKS.keys())
    usd_prices = get_price_series(df, CONSTS.CURRENCY, 'price')
    
    task1_results = []
    for symbol in stock_symbols:
        stock_prices = get_price_series(df, symbol, 'price')
        corr = helper.calculate_correlation(stock_prices.values, usd_prices.values)
        task1_results.append({'Stock': symbol, 'Sector': CONSTS.STOCKS[symbol], 'Correlation with USD/TRY': corr})
        print(f"  {symbol} ({CONSTS.STOCKS[symbol]}): {corr:.4f}")
    
    # Plot correlation matrix (stocks only)
    plot_correlation_matrix(df)
    
    # TASK 2: Correlation between stock prices in USD
    print("\n" + "="*60)
    print("TASK 2: Stock Prices (USD) Correlation Matrix")
    print("="*60)
    
    # Build USD price dataframe
    data_usd = {}
    for symbol in stock_symbols:
        data_usd[symbol] = get_price_series(df, symbol, 'price_usd')
    df_usd = pd.DataFrame(data_usd)
    
    # Print pairwise correlations
    n = len(stock_symbols)
    for i in range(n):
        for j in range(i+1, n):
            corr = helper.calculate_correlation(df_usd.iloc[:, i].values, df_usd.iloc[:, j].values)
            print(f"  {stock_symbols[i]} vs {stock_symbols[j]}: {corr:.4f}")
    
    plot_usd_correlation_matrix(df)

if __name__ == "__main__":
    main()