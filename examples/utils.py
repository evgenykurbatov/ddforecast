"""
"""

import math
import numpy as np

import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset):
    def __init__(self, series, input_len, forecast_horizon, dtype=torch.float32):
        self.series = series
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        self.dtype = dtype

    def __len__(self):
        return len(self.series) - self.input_len - self.forecast_horizon

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.input_len]
        y = self.series[idx+self.input_len:idx+self.input_len+self.forecast_horizon]
        return torch.tensor(x, dtype=self.dtype).unsqueeze(-1), \
               torch.tensor(y, dtype=self.dtype)


class MultiSeriesDataset(Dataset):

    def __init__(self, series_list, input_len, forecast_horizon, dtype=torch.float32):
        """
        series_list: list of 1D numpy arrays, each representing one time series
        """
        self.samples = []
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        self.dtype = dtype

        for series in series_list:
            series = np.asarray(series)
            for i in range(len(series) - input_len - forecast_horizon):
                x = series[i:i+input_len]
                y = series[i+input_len:i+input_len+forecast_horizon]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=self.dtype).unsqueeze(-1), \
               torch.tensor(y, dtype=self.dtype)


def validate_model(model, valid_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()

    return val_loss / len(valid_loader)


def train_model(model, train_loader, valid_loader, optimizer, criterion, device, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = validate_model(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, valid_loss={val_loss:.4f}")
