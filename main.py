import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import CONSTS
from helper import helper

def download_and_save_price_data() -> None:
    if not os.path.exists("prices"):
        os.makedirs("prices")

    stock_symbols = list(CONSTS.STOCKS.keys())
    for stock in stock_symbols:
        file_path = f"prices/{stock}.csv"
        if os.path.exists(file_path):
            continue
        df = yf.download(stock, start=CONSTS.START_DATE, end=CONSTS.END_DATE, progress=False)   
        with open(file_path, "w") as f:
            for i in range(len(df)):
                date = df.index[i].strftime('%Y-%m-%d')
                price = f"{df['Close'].values[i][0]:.3f}"
                f.write(f"{date},{price}\n") 
    
    currency_file = f"prices/{CONSTS.CURRENCY}.csv"
    if not os.path.exists(currency_file):
        currency = yf.download(CONSTS.CURRENCY, start=CONSTS.START_DATE, end=CONSTS.END_DATE, progress=False)
        with open(currency_file, "w") as f:
            for i in range(len(currency)):
                date = currency.index[i].strftime('%Y-%m-%d')
                price = f"{currency['Close'].values[i][0]:.3f}"
                f.write(f"{date},{price}\n")

def synchronize_data() -> None:
    stock_symbols = list(CONSTS.STOCKS.keys())
    all_symbols = stock_symbols + [CONSTS.CURRENCY]
    date_sets = []
    for symbol in all_symbols:
        file_path = f"prices/{symbol}.csv"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                dates = {line.split(",")[0] for line in f}
                date_sets.append(dates)
    
    if not date_sets: return
    common_dates = set.intersection(*date_sets)
    
    for symbol in all_symbols:
        file_path = f"prices/{symbol}.csv"
        if os.path.exists(file_path):
            lines_to_keep = []
            with open(file_path, "r") as f:
                for line in f:
                    if line.split(",")[0] in common_dates:
                        lines_to_keep.append(line)
            lines_to_keep.sort()
            with open(file_path, "w") as f:
                f.writelines(lines_to_keep)
    print(f"Data synchronized. Common dates: {len(common_dates)}")

def read_price_data(stock_symbol: str) -> pd.Series:
    file_path = f"prices/{stock_symbol}.csv"
    dates, prices = [], []
    with open(file_path, "r") as f:
        for line in f:
            date, price = line.strip().split(",")
            dates.append(date)
            prices.append(float(price))
    return pd.Series(data=prices, index=pd.to_datetime(dates))


def load_data() -> pd.DataFrame:
    """Load the main data.csv file."""
    return pd.read_csv("data.csv", parse_dates=['date'])


def get_price_series(df: pd.DataFrame, symbol: str, price_col: str = 'price') -> pd.Series:
    """Extract price series for a given symbol from the dataframe."""
    symbol_df = df[df['symbol'] == symbol].set_index('date').sort_index()
    return symbol_df[price_col]


def create_data_csv() -> None:
    stock_symbols = list(CONSTS.STOCKS.keys())
    usd_prices = read_price_data(CONSTS.CURRENCY)
    all_dfs = []

    # Process Stocks
    for symbol in stock_symbols:
        prices_try = read_price_data(symbol)
        prices_usd = prices_try / usd_prices
        
        df = pd.DataFrame({
            'symbol': symbol,
            'price': prices_try,
            'price_usd': prices_usd,
            'change_in_value_try': prices_try.diff(),
            'change_in_value_usd': prices_usd.diff(),
            'change_in_prcntg_try': prices_try.pct_change(),
            'change_in_prcntg_usd': prices_usd.pct_change()
        })
        all_dfs.append(df)

    # Process Currency
    df_curr = pd.DataFrame({
        'symbol': CONSTS.CURRENCY,
        'price': usd_prices,
        'price_usd': 1.0,
        'change_in_value_try': usd_prices.diff(),
        'change_in_value_usd': 0.0,
        'change_in_prcntg_try': usd_prices.pct_change(),
        'change_in_prcntg_usd': 0.0
    })
    all_dfs.append(df_curr)

    final_df = pd.concat(all_dfs).reset_index().rename(columns={'index': 'date'})
    final_df.to_csv("data.csv", index=False)
    print("data.csv created successfully.")

def plot_correlation_matrix() -> None:
    """Plot correlation matrix for stock prices in TRY using data.csv."""
    if not os.path.exists("img"):
        os.makedirs("img")
    
    df = load_data()
    stock_symbols = list(CONSTS.STOCKS.keys())
    
    # Build a DataFrame with prices for each symbol
    data = {}
    for symbol in stock_symbols:
        data[symbol] = get_price_series(df, symbol, 'price')
    
    price_df = pd.DataFrame(data)
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
    plt.title("Stock Correlation Matrix")
    plt.tight_layout()
    plt.savefig("img/correlation_matrix.png")
    plt.close()


def plot_usd_correlation_matrix() -> None:
    """Plot correlation matrix for stock prices in USD using data.csv."""
    if not os.path.exists("img"):
        os.makedirs("img")
    
    df = load_data()
    stock_symbols = list(CONSTS.STOCKS.keys())
    
    # Build a DataFrame with USD prices for each symbol
    data_usd = {}
    for symbol in stock_symbols:
        data_usd[symbol] = get_price_series(df, symbol, 'price_usd')
    
    df_usd = pd.DataFrame(data_usd)
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
    plt.savefig("img/usd_correlation_matrix.png")
    plt.close()

def main():
    download_and_save_price_data()
    synchronize_data()
    create_data_csv()
    plot_correlation_matrix()
    plot_usd_correlation_matrix()

if __name__ == "__main__":
    main()