import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fft
import pywt

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

def fourier_denoise(x, threshold = 0.0, keep_ratio = 0.0):
    """
    Remove the frequency with amplitude smaller than threshold or keep the largest FFT coefs. 
    If there are keep_ratio and threshold at the same time, keep the largest FFT coef that 
    higher than threshold
    Args:
        x (array-like): Input signal to be filtered.
        fs (float): Sampling frequency of the input signal.
        threshold (float): Cutoff amplitude of frequency.
        keep_ratio (float): Percentage of coef used for denoise
    Returns:
        np.ndarray: denoised signal.
    """
    x = np.asarray(x)
    xfft = np.fft.fft(x) / len(x)

    # remove the coef that are smaller than threshold
    if threshold > 0:
        mask = np.abs(xfft) < threshold
        xfft[mask] = 0

    # Keep only the largest coef
    if keep_ratio > 0:
        k = int(np.ceil(keep_ratio * len(x)))
        thresh = np.partition(np.abs(xfft), -k)[-k]  # kth largest magnitude
        xfft = np.where(np.abs(xfft) >= thresh, xfft, 0)
    filtered_signal = np.fft.ifft(xfft) * len(x)
    
    return np.real(filtered_signal)

def wavelet_denoise(x, keep_ratio=0.0, threshold = 0.0, wavelet="db4", level=1, soft_thresholding=False,
                    verbose = True):
    """
    Remove the coef with amplitude smaller than threshold or keep the largest wavelet coefs. 
    If there is keep_ratio and threshold at the same time, keep the largest wavelet coef that 
    higher than threshold. 
    If soft_thresholding = True, the threshold value is estimated using the result of Donoho and Johnson(1994)
    if both threshold and soft_thresholding are given, the higher value will be chosen
    Args:
        x (array-like): Input signal to be filtered.
        fs (float): Sampling frequency of the input signal.
        threshold (float): Cutoff amplitude of frequency.
        keep_ratio (float): Percentage of coef used for denoise
        soft_thresholding(bool) : To use the soft thresholding
        verbose(bool): To print the threshold and keep_ratio
    Returns:
        np.ndarray: denoised signal.
    """
    x = np.asarray(x)
    L = min(pywt.dwt_max_level(len(x), wavelet),level)
    xwav = pywt.wavedecn(data=x, wavelet=wavelet, mode="per", level=L)
    arr, coef_slices = pywt.coeffs_to_array(xwav)
    arr_flat = arr.flatten()

    if soft_thresholding :
        cD1 = xwav[-1]['d']
        sigma = np.median(np.abs(cD1)) / 0.6745
        lam = sigma * np.sqrt(2 * np.log(len(x)))

        # compare threshold and the estimated threshold value
        threshold = max(threshold, lam)

    if threshold > 0:
        mask = np.abs(arr_flat) < threshold
        arr_flat[mask] = 0

    # Keep only the largest coef
    if keep_ratio > 0:
        k = int(np.ceil(keep_ratio * len(arr_flat)))
        thresh = np.partition(np.abs(arr_flat), -k)[-k]  # kth largest magnitude
        arr_flat = np.where(np.abs(arr_flat) >= thresh, arr_flat, 0)

    coef_from_arr = pywt.array_to_coeffs(arr_flat, coef_slices)
    Srec = pywt.waverecn(coef_from_arr, wavelet, mode="per")

    if verbose:
        if threshold > 0 and keep_ratio > 0:
            print(f"threshold = {threshold}, keep ratio = {keep_ratio}")
        elif threshold == 0 and keep_ratio > 0:
            print(f"keep ratio = {keep_ratio}")
        elif threshold > 0 and keep_ratio == 0:
            print(f"magnitude = {threshold}")

    return Srec

def wavelet_plot_coef(x, wavelet="db4", level=1):
    x = np.asarray(x)
    L = min(pywt.dwt_max_level(len(x), wavelet),level)
    xwav = pywt.wavedecn(data=x, wavelet=wavelet, mode="per", level=L)
    arr, coef_slices = pywt.coeffs_to_array(xwav)
    arr = np.abs(arr.flatten())
    arr.sort()

    plt.plot(arr[::-1])
    plt.grid()
    plt.title(f"Magnitude of wavelet coefficients, level={L}")
    plt.show()


def wavelet_plot_mra(x, wavelet="db4", level=1):
    """
    Plot a wavelet multiresolution analysis (MRA) like the classic D1..Dn, S_n stacked figure.
    x: 1D array
    wavelet: e.g. 'db4', 'sym5', 'coif1', 'haar'
    level: decomposition level 
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    L = min(pywt.dwt_max_level(N, wavelet),level)

    # Decompose
    coeffs = pywt.wavedec(x, wavelet, level=L)

    # Reconstruct approximation S_level and each detail D_j to full length
    # coeffs format: [cA_level, cD_level, cD_{level-1}, ..., cD1]
    A = pywt.upcoef('a', coeffs[0], wavelet, level=L, take=N)

    Ds = []
    for j in range(L, 0, -1):
        # detail coefficients are coeffs[level - j + 1] because of ordering
        cDj = coeffs[L - j + 1]
        Dj = pywt.upcoef('d', cDj, wavelet, level=j, take=N)
        Ds.append(Dj)  # Ds[0] is D_level, last is D1

    # For display like the figure: show D1 at top, ..., D_level, then S_level, then Data
    Ds_display = Ds[::-1]  # now [D1, D2, ..., D_level]

    nrows = L + 2  # D1..D_level + S + Data
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 1.2*nrows), sharex=True)
    fig.suptitle("Multiresolution analysis", y=0.995)

    # Plot details
    for i, Dj in enumerate(Ds_display, start=1):
        ax = axes[i-1]
        ax.plot(Dj, linewidth=1)
        ax.set_ylabel(f"D{i}", rotation=0, labelpad=20)
        ax.axhline(0, linewidth=0.5)
        ax.set_yticks([])

    # Plot approximation
    axS = axes[L]
    axS.plot(A, linewidth=1)
    axS.set_ylabel(f"S{L}", rotation=0, labelpad=20)
    axS.axhline(0, linewidth=0.5)
    axS.set_yticks([])

    # Plot original data
    axX = axes[L+1]
    axX.plot(x, linewidth=1)
    axX.set_ylabel("Data", rotation=0, labelpad=20)
    ax.axhline(0, linewidth=0.5)
    axX.set_yticks([])
    axX.set_xlabel("Time (samples)")

    plt.tight_layout()
    plt.show()