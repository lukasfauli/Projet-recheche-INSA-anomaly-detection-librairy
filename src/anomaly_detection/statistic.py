import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def zscore_glissant(window_size, df_column, verbose=True, threshold=3):
    """
    Detect anomalies in a time series using a sliding Z-Score method.
    
    This function calculates the rolling mean and standard deviation to define 
    dynamic thresholds (a 'tunnel of normality'). Values falling outside this 
    tunnel are marked as outliers.

    Args:
        window_size (int): The number of points to include in the rolling window.
        df_column (pd.Series): The input data column to analyze.
        verbose (bool): If True, displays a plot of the signal, thresholds, and detected outliers.
        threshold (int or float): The multiplier for the standard deviation (k) to set the 
            boundary. Defaults to 3 (standard statistical threshold).

    Returns:
        pd.Series: A boolean vector (True for outliers, False otherwise) of the 
            same length as the input column.
            Note: The first (window_size - 1) values will be False as the 
            rolling statistics are not yet computed (NaN).
    """
    
    rolling_mean = df_column.rolling(window=window_size).mean()
    rolling_std = df_column.rolling(window=window_size).std()

    upper_bound = rolling_mean + (threshold * rolling_std)
    lower_bound = rolling_mean - (threshold * rolling_std)

    # Note: NaN values in bounds result in False during comparison
    is_outlier = (df_column > upper_bound) | (df_column < lower_bound)

    if verbose:
        plt.figure(figsize=(15, 7))
        x = df_column.index

        plt.plot(x, df_column, color='gray', alpha=0.5, label='Signal')
        plt.plot(x, rolling_mean, color='blue', alpha=0.8, label='Moving Average')
        
        plt.plot(x, upper_bound, 'r--', alpha=0.8, label=f'Boundaries ({threshold}σ)')
        plt.plot(x, lower_bound, 'r--', alpha=0.8)
        
        plt.scatter(x[is_outlier], df_column[is_outlier], color='red', label='Outliers')
        
        plt.title(f"Anomaly Detection on {df_column.name} (Window: {window_size})")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    return is_outlier

def detect_constant(signal, tol=0.01, min_length=10, verbose=False, plot = False):
    signal = np.asarray(signal)
    grad = np.abs(np.diff(signal))
    idx = None
    
    for i in range(len(grad) - min_length):
        if np.nanmax(grad[i:]) < tol:
            idx = i
            break
    
    if verbose:
        if idx is not None:
            print(f"Signal becomes constant at index {idx}")
        else:
            print("No constant region detected.")
            
    if plot:
        plt.figure()
        plt.plot(signal)
        
        if idx is not None:
            plt.axvline(idx, linestyle='--')
            plt.scatter(idx, signal[idx])
        
        plt.title("Signal with Detected Constant Region")
        plt.xlabel("Index")
        plt.ylabel("Signal value")
        plt.show()
    
    return idx