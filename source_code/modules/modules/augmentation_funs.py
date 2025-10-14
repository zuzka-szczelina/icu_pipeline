import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


def fourier_augment_dc(time_series: np.ndarray | pd.core.series.Series,
                       noise_factor: float = 0.1) -> np.ndarray:

    freq_spectrum = np.fft.fft(time_series)
    original_dc_component = freq_spectrum[0]

    # Apply scaling and noise to all components
    noise = np.random.normal(size=freq_spectrum.shape) * noise_factor
    modified_freq_spectrum = freq_spectrum + noise

    # Restore unscaled DC component
    modified_freq_spectrum[0] = original_dc_component

    # augmented_time_series = np.fft.ifft(modified_freq_spectrum).real - scale_factor + 1
    augmented_time_series = np.fft.ifft(modified_freq_spectrum).real
    return augmented_time_series

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