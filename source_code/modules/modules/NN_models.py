import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


class LSTMAttentionForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2,
                 output_dim=1, dropout=0.2):
        super(LSTMAttentionForecaster, self).__init__()
        # LSTM Layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        # Attention Layer
        self.attention = nn.Linear(hidden_dim, 1)

        # Fully Connected Layers for Output
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention Mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully Connected Layers
        x = self.dropout(self.relu(self.fc1(context_vector)))
        output = self.fc2(x)
        return output
        