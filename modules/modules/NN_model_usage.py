import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from .NN_data_prep import plot_time_series
from .NN_models import LSTMForecaster, LSTMAttentionForecaster


def train_model(model, X, Y,  training_series_length=None, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X[0:training_series_length])
        if type(model) == LSTMForecaster:
            loss = criterion(pred[:, -1, :], Y[0:training_series_length])
        elif type(model) == LSTMAttentionForecaster:
            loss = criterion(pred, Y[0:training_series_length])
        loss.backward()
        optimizer.step()
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item()}")

def model_predict(model, X):
    model.eval()
    with torch.no_grad():
        if type(model) == LSTMForecaster:
            if len(X) > 1:
                pred = model(X).numpy().squeeze()[:, -1]
            elif len(X) == 1:
                pred = model(X).numpy().squeeze()[-1]
        elif type(model) == LSTMAttentionForecaster:
            pred = model(X).numpy().squeeze()
    return pred

def detect_anomalies(pred, Y, seq_len, threshold_scaler=1):
    # if type(model) == LSTMForecaster:
    #     residuals = np.abs(Y.numpy().squeeze() - pred[:, -1])
    # elif type(model) == LSTMAttentionForecaster:
    #     residuals = np.abs(Y.numpy().squeeze() - pred.squeeze())
    residuals = np.abs(Y.numpy().squeeze() - pred.squeeze())
    threshold = np.mean(residuals) + 2 * np.std(residuals)
    threshold *= threshold_scaler
    detected_anomalies = np.where(residuals > threshold)[0] + seq_len
    return detected_anomalies

def create_saving_path(
    model,
    seq_len,
    series_name,
    folder='',
    subfolder=''):
    if type(model) == LSTMForecaster:
        model_folder_name = 'LSTMForecaster'
    elif type(model) == LSTMAttentionForecaster:
        model_folder_name = 'LSTMAttentionForecaster'
    # filename = 'pred_seg_len_' + str(seq_len) + '.png'
    filename = f'pred_seg_len_{seq_len:02d}' + '.png'
    path_to_file = os.path.join(
        '../images', 
        folder,
        subfolder,
        series_name, 
        model_folder_name)
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
    saving_path = os.path.join(path_to_file, filename)
    return saving_path
    
def plot_results(
    x,
    y,
    predictions,
    anomalies,
    detected_anomalies,
    training_series_length,
    seq_len,
    saving_path=None,
    finetuning=False,
    finetuning_step=None):
    # plt.figure()
    plot_time_series(x, y, anomalies, predictions, title=f"Time Series with Predictions and Anomalies, seq_len={seq_len}")
    plt.scatter(x[detected_anomalies], y[detected_anomalies], color='orange',  label="Detected Anomalies")
    # plt.vlines(x[training_series_length], -1, 1, color='y', label='end of training series')
    plt.vlines(x[training_series_length], -1, 1, color='y', label='end of initial training')

    if finetuning:
        finetuning_ends = list(range(training_series_length + finetuning_step, len(x), finetuning_step))
        plt.vlines(x[finetuning_ends], -1, 1, colors='y', linestyles='dashed', label='finetuning moments')
    
    
    plt.legend(loc='upper right')
    if saving_path is not None:
        plt.savefig(saving_path)
    plt.show()

