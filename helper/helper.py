import numpy as np
import pandas as pd

def calculate_covariance(data_1: pd.Series, data_2: pd.Series) -> float:
    if len(data_1) != len(data_2):
        raise ValueError("Data series must be of the same length to calculate covariance.")
    
    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)
    covariance = np.sum((data_1 - mean_1) * (data_2 - mean_2)) / (len(data_1) - 1)
    return covariance

def calculate_variance(data: pd.Series) -> float:
    mean = np.mean(data)
    variance = np.sum((data - mean) ** 2) / (len(data) - 1)
    return variance
    
def calculate_std_dev(data: pd.Series) -> float:
    variance = calculate_variance(data)
    std_dev = np.sqrt(variance)
    return std_dev

def calculate_correlation(data_1: pd.Series, data_2: pd.Series) -> float:
    covariance = calculate_covariance(data_1, data_2)
    std_dev_1 = calculate_std_dev(data_1)
    std_dev_2 = calculate_std_dev(data_2)
    
    if std_dev_1 == 0 or std_dev_2 == 0:
        return 0.0
        
    correlation = covariance / (std_dev_1 * std_dev_2)
    return correlation

