import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# def generate_time_series(num_samples=200):
#     x = np.arange(0, num_samples)
#     y = np.sin(x * 0.1) + np.random.normal(scale=0.1, size=num_samples)
#     anomalies = np.random.choice(np.arange(20, num_samples-20), size=5, replace=False)
#     y[anomalies] += np.random.normal(scale=2.0, size=5)  # Add anomalies
#     return x, y, anomalies

def preprocess_series(series):
    x = np.array(series.index)
    y = np.array(series.values)
    y = 2  * (y - min(y)) / (max(y) - min(y)) -1
    return x, y

def plot_time_series( x, y, anomalies=[], predictions=None, title="Time Series"):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="True Data")

    if predictions is None:
        pred_start = 0
    else:
        pred_start = len(x) - len(predictions)
    if anomalies != []:
        plt.scatter(x[anomalies], y[anomalies], s=90, color='red', alpha=1, label="Anomalies")
    if predictions is not None:
        plt.plot(x[pred_start:], predictions, label="Predictions", linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Value (normalised)")
    plt.title(title)
    plt.legend()

def prepare_dataset(y, seq_len=20):
    X, Y = [], []
    for i in range(len(y) - seq_len):
        X.append(y[i:i+seq_len])
        Y.append(y[i+seq_len])
    return torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1), torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

