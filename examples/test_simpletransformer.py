# %%

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt

import context
from context import ddforecast
import utils

# %%
# Load data

dtype = torch.float32

seq = torch.load('data-gp.pt')
print(seq.shape)

train_seq = seq[:20]
valid_seq = seq[20:]

input_len = 30
forecast_horizon = 1
train_dataset = utils.MultiSeriesDataset(train_seq, input_len, forecast_horizon, dtype)
valid_dataset = utils.MultiSeriesDataset(valid_seq, input_len, forecast_horizon, dtype)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64)

# %%
# Initialise model and loss criterion

model = ddforecast.SimpleTransformer(input_dim=1, forecast_horizon=forecast_horizon)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()

# %%
# Train

optimizer = optim.Adam(model.parameters(), lr=1e-3)
utils.train_model(model, train_loader, valid_loader, optimizer, criterion, device, epochs=20)

# %%

# Save the model
torch.save(model.state_dict(), 'test_simpletransformer.model')

# %%

def plot_predictions(model, dataset, num_samples=100):
    model.eval()
    inputs = []
    targets = []
    predictions = []

    with torch.no_grad():
        for i in range(num_samples):
            x, y = dataset[i]
            x_input = x.unsqueeze(0).to(device)  # [1, input_len, 1]
            pred = model(x_input)  # [1, forecast_horizon]

            inputs.append(x.squeeze().cpu().numpy())
            targets.append(y.squeeze().cpu().numpy())
            predictions.append(pred.squeeze().cpu().numpy())

    # Plot results
    input_len        = len(dataset[0][0])
    forecast_horizon = len(dataset[0][1])
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(num_samples):
        start = i * (forecast_horizon)
        input_plot = np.arange(start, start + input_len)
        target_plot = np.arange(start + input_len, start + input_len + forecast_horizon)

        ax.plot(input_plot, inputs[i], color='blue', alpha=0.3)
        ax.plot(target_plot, targets[i], 'o', color='green', label='True' if i == 0 else "", alpha=0.6)
        ax.plot(target_plot, predictions[i], 'o', color='red', linestyle='--', label='Predicted' if i == 0 else "", alpha=0.6)

    ax.set_title(f'Time Series Forecasting ({num_samples} samples)')
    ax.legend()
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Call it
plot_predictions(model, valid_dataset, num_samples=500)

# %%
