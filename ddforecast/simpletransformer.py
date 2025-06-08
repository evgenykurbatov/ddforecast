"""
2025-06-07 on chatgpt.com
using pytorch, make a transformer for forecasting of a sequence of scalar (not categorical) values

```
Here's a simple PyTorch implementation of a Transformer-based model for time series forecasting of scalar (continuous) values.

This model uses the Transformer encoder for processing past observations to predict future values.
It's ideal for 1D scalar time series like temperature, stock prices, etc.
```
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, base=10000):
        super().__init__()
        assert base > max_len
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)  # Even indices
        pe[:,1::2] = torch.cos(position * div_term)  # Odd indices
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class SimpleTransformer(nn.Module):

    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=3, dropout=0.1, forecast_horizon=1):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, activation='gelu', dropout=dropout, batch_first=True)
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, forecast_horizon)

    def forward(self, x):
        # x: [batch_size, sequence_length, 1]
        x = self.input_proj(x)  # [batch_size, seq_len, model_dim]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)  # [batch_size, seq_len, model_dim]
        x = x[:,-1,:]  # Take the last time step's output
        output = self.decoder(x)  # [batch_size, forecast_horizon]
        return output
