# %%

import time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt

import context
from context import ddforecast
from ddforecast.simpletransformer import *
import utils

# %%
# Load data

dtype = torch.float32

seq = torch.load(context.tmp_dir / 'data-gp.pt')
print(seq.shape)

train_seq = seq[:100]
valid_seq = seq[100:]

input_len = 30
forecast_horizon = 1
train_dataset = utils.MultiSeriesDataset(train_seq, input_len, forecast_horizon, dtype)
valid_dataset = utils.MultiSeriesDataset(valid_seq, input_len, forecast_horizon, dtype)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64)

# %%
# Initialise model and loss criterion

#model = SimpleTransformer(input_dim=1, forecast_horizon=forecast_horizon)
model = SimpleTransformer(input_dim=1, model_dim=16, num_heads=2, forecast_horizon=forecast_horizon)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()

# %%
# Load or train the model

force_train = False
#force_train = True

fname = context.tmp_dir / 'test_simpletransformer.model'
if fname.is_file() and not force_train:
    print("Load model")
    model.load_state_dict(torch.load(fname, weights_only=True))
    model.to(device)
else:
    print("Train model")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    utils.train_model(model, train_loader, valid_loader, optimizer, criterion, device, epochs=20)
    print("Elapsed time:", time.time() - t0)
    print("\nSave model")
    torch.save(model.state_dict(), fname)

# %%

def plot_predictions(model, series, input_len, forecast_horizon=1):
    #
    # Evaluate predictions

    model.eval()
    inputs, targets, preds, diffs = [], [], [], []

    with torch.no_grad():
        for i in range(len(series) - input_len - forecast_horizon):
            x = series[i:i+input_len]
            y = series[i+input_len:i+input_len+forecast_horizon]

            x_input = x.unsqueeze(0).unsqueeze(-1).to(device)  # [1, input_len, 1]
            y_pred = model(x_input).squeeze()  # [1, forecast_horizon]

            inputs.append(x.squeeze().cpu().numpy())
            targets.append(y.squeeze().cpu().numpy())
            preds.append(y_pred.squeeze().cpu().numpy())
            diffs.append(preds[-1] - targets[-1])

    #
    # Plot predictions

    num_samples = len(series) - input_len - forecast_horizon
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(12, 4))

    ax_ = ax[0]
    ax_.set_title(f'Time Series Forecasting ({num_samples} samples)')
    for i in range(num_samples):
        start = i * forecast_horizon
        input_plot = np.arange(start, start + input_len)
        target_plot = np.arange(start + input_len, start + input_len + forecast_horizon)
        ax_.plot(input_plot,  inputs[i], color='blue', alpha=0.3)
        ax_.plot(target_plot, targets[i], 'o', color='green', label='True' if i == 0 else "", alpha=0.6)
        ax_.plot(target_plot, preds[i], 'o', color='red', linestyle='--', label='Predicted' if i == 0 else "", alpha=0.6)
    ax_.legend()
    ax_.set_xlabel('Time step')
    ax_.set_ylabel('Value')
    ax_.grid(True)

    ax_ = ax[1]
    ax_.hist(np.array(diffs).T)

    return fig


from matplotlib.backends.backend_pdf import PdfPages

# For multipage PDF see https://matplotlib.org/stable/gallery/misc/multipage_pdf.html
with PdfPages(context.tmp_dir / 'test_simpletransformer.pdf') as pdf:
    for seq_ in tqdm(valid_seq):
        fig = plot_predictions(model, seq_, input_len, forecast_horizon)
        pdf.savefig(fig)
        plt.close()

# %%
