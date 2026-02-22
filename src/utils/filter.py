import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fft

def moving_average(x, w = 10):
    """
    Simple moving average filter
    Args:
        x (array-like): Input signal to be smoothed.
        w (int): Window size for the moving average. Default is 10.
    Returns:
        np.ndarray: Smoothed signal after applying the moving average filter.
    """
    return signal.convolve(np.asarray(x), np.ones(w), 'same') / w

def fourier_plot(x, fs=1):
    """
    Plot Real, Imaginary, Magnitude and Phase of FFT.
    
    Parameters
    ----------
    x : array-like
        Input signal
    fs : float
        Sampling frequency (default = 1)
    """
    
    x = np.asarray(x)
    N = len(x)
    
    # FFT with normalization
    xfft = fft.fft(x) / N
    freqs = fft.fftfreq(N, d=1/fs)
    
    plt.figure(figsize=(12, 8))
    
    # Real
    plt.subplot(2, 2, 1)
    plt.plot(freqs, xfft.real)
    plt.title("Real Part")
    plt.xlabel("Frequency")
    plt.grid()
    
    # Imaginary
    plt.subplot(2, 2, 2)
    plt.plot(freqs, xfft.imag)
    plt.title("Imaginary Part")
    plt.xlabel("Frequency")
    plt.grid()
    
    # Magnitude
    plt.subplot(2, 2, 3)
    plt.plot(freqs, np.abs(xfft))
    plt.title("Magnitude")
    plt.xlabel("Frequency")
    plt.grid()
    
    # Phase
    plt.subplot(2, 2, 4)
    plt.plot(freqs, np.angle(xfft))
    plt.title("Phase")
    plt.xlabel("Frequency")
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def fourier_low_pass(x, fs, cutoff):
    """
    Apply a low-pass Fourier filter to the input signal.
    Args:
        x (array-like): Input signal to be filtered.
        fs (float): Sampling frequency of the input signal.
        cutoff (float): Cutoff frequency for the low-pass filter.
    Returns:
        np.ndarray: Filtered signal after applying the low-pass Fourier filter.
    """
    # Compute the Fourier Transform of the input signal
    x = np.asarray(x)
    xfft = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1/fs)

    # Create a mask for frequencies above the cutoff
    mask = np.abs(freqs) > cutoff
    xfft[mask] = 0  # Zero out frequencies above the cutoff

    # Compute the Inverse Fourier Transform to get the filtered signal
    filtered_signal = np.fft.ifft(xfft)
    
    return np.real(filtered_signal)

def fourier_denoise(x, magnitude = 0.0, keep_ratio = 0.0):
    """
    Remove the frequency with amplitude smaller than magnitude or keep the largest FFT coefs. 
    If there is keep_ratio and magnitude at the same time, keep the largest FFT coef that 
    higher than magnitude, 
    Args:
        x (array-like): Input signal to be filtered.
        fs (float): Sampling frequency of the input signal.
        magnitude (float): Cutoff amplitude of frequency.
        keep_ratio (float): Percentage of coef used for denoise
    Returns:
        np.ndarray: denoised signal.
    """
    x = np.asarray(x)
    xfft = np.fft.fft(x) / len(x)

    # remove the coef that are smaller than magnitude
    mask = np.abs(xfft) < magnitude
    xfft[mask] = 0

    # Keep only the largest coef
    k = int(np.ceil(keep_ratio * len(x)))
    thresh = np.partition(np.abs(xfft), -k)[-k]  # kth largest magnitude
    xfft = np.where(np.abs(xfft) >= thresh, xfft, 0)
    filtered_signal = np.fft.ifft(xfft) * len(x)
    
    return np.real(filtered_signal)
