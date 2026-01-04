import datetime

STOCKS = {
    # Bank
    'GARAN.IS': 'Garanti BBVA',
    'AKBNK.IS': 'Akbank',
    # Energy
    'ENJSA.IS': 'EnerjiSA',
    'ZOREN.IS': 'Zorlu',
    # Food
    'BIMAS.IS': 'BİM',
    'ULKER.IS': 'Ülker',
}

CURRENCY = 'USDTRY=X'

START_DATE = '2021-01-01'
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')