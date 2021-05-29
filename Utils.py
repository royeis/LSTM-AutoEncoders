import torch
from torch.utils.data.dataset import TensorDataset
import random
import numpy as np
import matplotlib.pyplot as plt

def prepare_tensor_dataset(dataset):
    return TensorDataset(torch.Tensor(dataset))

def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_signals(signals):
    signals = signals.squeeze()
    seq_len = len(signals[0])
    fig, axs = plt.subplots(len(signals))
    fig.suptitle('Signals')
    for sig, ax in zip(signals, axs):
        ax.plot(range(seq_len), sig)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.show()


def plot_signals_with_rec(signals, signals_rec):
    signals, signals_rec = signals.squeeze(), signals_rec.squeeze()
    seq_len = len(signals[0])
    fig, axs = plt.subplots(len(signals))
    fig.suptitle('Original vs Random Generated Signals')
    for sig, sig_rec, ax in zip(signals, signals_rec, axs):
        ax.plot(range(seq_len), sig, label="Original Signal")
        ax.plot(range(seq_len), sig_rec, label="Reconstructed Signal")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()
    fig.show()
