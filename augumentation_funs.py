import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


# """version 1 - unused"""
def fourier_augment(time_series, noise_factor=0.1, scale_factor=1.0):
    freq_spectrum = np.fft.fft(time_series)
    scaled_freq_spectrum = freq_spectrum * scale_factor
    noise = np.random.normal(size=freq_spectrum.shape) * noise_factor
    modified_freq_spectrum = scaled_freq_spectrum + noise
    augmented_time_series = np.fft.ifft(modified_freq_spectrum).real
    return augmented_time_series

def fourier_augment_dc(time_series: np.ndarray | pd.core.series.Series,
                       noise_factor: float = 0.1,
                       scale_factor: float = 1.0) -> np.ndarray:

    freq_spectrum = np.fft.fft(time_series)
    original_dc_component = freq_spectrum[0]

    # Apply scaling and noise to all components
    scaled_freq_spectrum = freq_spectrum * scale_factor
    noise = np.random.normal(size=freq_spectrum.shape) * noise_factor
    modified_freq_spectrum = scaled_freq_spectrum + noise

    # Restore unscaled DC component
    modified_freq_spectrum[0] = original_dc_component

    # augmented_time_series = np.fft.ifft(modified_freq_spectrum).real - scale_factor + 1
    augmented_time_series = np.fft.ifft(modified_freq_spectrum).real
    return augmented_time_series

"""plots:"""
""" simple compare - statistical based """
def statistical_comparison(original, augmented_list):
    for i, augmented in enumerate(augmented_list, 1):
        original_mean = np.mean(original)
        augmented_mean = np.mean(augmented)
        original_variance = np.var(original)
        augmented_variance = np.var(augmented)
        print(f"Augmented Series {i}:")
        print(f"Original Mean: {original_mean}, Augmented Mean: {augmented_mean}")
        print(f"Original Variance: {original_variance}, Augmented Variance: {augmented_variance}")
        print()

""" augmented vs. true - superposition """
def plot_overlay_multiple(original, augmented_list, time, saving_path):
    plt.figure(figsize=(12, 6))
    for i, augmented in enumerate(augmented_list, 1):
        plt.plot(time, augmented, label=f"Augmented Series {i}", alpha=0.6)
    plt.plot(time, original, label="Original Series", color="black", linewidth=2, alpha=0.8)
    # plt.legend()
    plt.xlabel("Time")
    plt.ylabel(original.name)
    plt.title("Overlay Plot of Original and Multiple Augmented Series")
    plt.savefig(saving_path)
    plt.show()

""" compare the augmented and original frequencies """
def frequency_spectrum_comparison_multiple(original, augmented_list):
    original_freq_spectrum = np.abs(np.fft.fft(original))
    plt.figure(figsize=(12, 6))
    plt.plot(original_freq_spectrum, label="Original Spectrum", color="black", linewidth=2)
    for i, augmented in enumerate(augmented_list, 1):
        scaled_freq_spectrum = np.abs(np.fft.fft(augmented))
        plt.plot(scaled_freq_spectrum, label=f"Augmented Spectrum {i}", linestyle="--", alpha=0.7)
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Frequency Spectrum Comparison")
    plt.show()

""" simple correlation plot """
def correlation_plot_multiple(original, augmented_list):
    plt.figure(figsize=(8, 6))
    for i, augmented in enumerate(augmented_list, 1):
        plt.scatter(original, augmented, alpha=0.5, label=f"Augmented Series {i}")
    plt.xlabel("Original Series")
    plt.ylabel("Augmented Series")
    plt.title("Correlation Plot between Original and All Augmented Series")
    plt.plot([min(original), max(original)], [min(original), max(original)], 'r--')  # 1:1 line
    plt.legend()
    plt.show()

""" residual plot - select the best augmented series """
def residual_plot_multiple(original, augmented_list, time):
    plt.figure(figsize=(12, 6))
    for i, augmented in enumerate(augmented_list, 1):
        residuals = (original - augmented)
        plt.plot(time, residuals, label=f"Residual for Augmented Series {i}", alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Residual (Original - Augmented)")
    plt.title("Residual Plot between Original and All Augmented Series")
    plt.legend()
    plt.show()

def relative_residual_plot_multiple(original, augmented_list, time):
    plt.figure(figsize=(12, 6))
    for i, augmented in enumerate(augmented_list, 1):
        residuals = (original - augmented)/(original)
        plt.plot(time, residuals, label=f"Residual for Augmented Series {i}", alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Residual relative value (Original - Augmented)")
    plt.title("Residual relative value Plot between Original and All Augmented Series")
    plt.legend()
    plt.show()