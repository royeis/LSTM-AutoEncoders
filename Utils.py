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
    plt.savefig('3_sig_example.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    fig.show()


def plot_signals_with_rec(signals, signals_rec, args):
    signals, signals_rec = signals.squeeze(), signals_rec.squeeze()
    seq_len = len(signals[0])
    fig, axs = plt.subplots(len(signals))
    fig.suptitle('Original vs RGS\n lr={:.4f} bsz={} hsz={}'
                 .format(args.lr, args.batch_size, args.hidden_size))
    for sig, sig_rec, ax in zip(signals, signals_rec, axs):
        ax.plot(range(seq_len), sig, label="Original Signal")
        ax.plot(range(seq_len), sig_rec, label="Reconstructed Signal")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()
    plt.savefig('3_sig_with_rec_example.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    fig.show()


def plot_stocks_with_rec(stock, stock_rec, seq_len,
                         fig_title='Original vs Reconstructed Stock Price',
                         label_1='Original Stock Value',
                         label_2='Reconstructed Stock Value',
                         epoch=0):
    signals, signals_rec = stock.squeeze(), stock_rec.squeeze()
    fig, axs = plt.subplots(len(signals))
    fig.suptitle(fig_title)
    for sig, sig_rec, ax, idx in zip(signals, signals_rec, axs, seq_len):
        ax.plot(range(idx), sig[:idx], label=label_1)
        ax.plot(range(idx), sig_rec[:idx], label=label_2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()
    plt.savefig('./figures/{}_epoch_{}.png'.format(fig_title, epoch),
                transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    fig.show()

def plot_err_epoch(train_loss, test_loss, train_acc, test_acc, epochs, suff, data=""):
    fig, ax = plt.subplots()
    ax.set_title(data +
                 '\nLoss vs. Epoch' +
                 f'\n{suff}')
    ax.plot(range(epochs), train_loss, label='Train Loss')
    ax.plot(range(epochs), test_loss, label='Test Loss')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.savefig('{}_loss_{}.png'.format(data, suff),
                transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(data +
                 '\nAccuracy vs. Epoch' +
                 f'\n{suff}')
    ax.plot(range(epochs), train_acc, label='Train Acc')
    ax.plot(range(epochs), test_acc, label='Test Acc')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.savefig('{}_acc_{}.png'.format(data, suff),
                transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


def plot_loss(train_loss, test_loss, epochs, suff, data="", val_loss=None):
    fig, ax = plt.subplots()
    ax.set_title(data +
                 '\nLoss vs. Epoch' +
                 f'\n{suff}')
    ax.plot(range(epochs), train_loss, label='Train Loss')
    if val_loss is not None:
        ax.plot(range(epochs), val_loss, label='Validation Loss')
    ax.plot(range(epochs), test_loss, label='Test Loss')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.savefig('{}_loss_{}.png'.format(data, suff),
                transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


def plot_err_epoch(train_loss, test_loss, train_acc, test_acc, epochs, suff, data=""):
    fig, ax = plt.subplots()
    ax.set_title(data +
                 '\nLoss vs. Epoch' +
                 f'\n{suff}')
    ax.plot(range(epochs), train_loss, label='Train Loss')
    ax.plot(range(epochs), test_loss, label='Test Loss')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.savefig('{}_loss_{}.png'.format(data, suff),
                transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(data +
                 '\nAccuracy vs. Epoch' +
                 f'\n{suff}')
    ax.plot(range(epochs), train_acc, label='Train Acc')
    ax.plot(range(epochs), test_acc, label='Test Acc')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.savefig('{}_acc_{}.png'.format(data, suff),
                transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


def plot_loss(train_loss, val_loss, test_loss, epochs, suff, data=""):
    fig, ax = plt.subplots()
    ax.set_title(data +
                 '\nLoss vs. Epoch' +
                 f'\n{suff}')
    ax.plot(range(epochs), train_loss, label='Train Loss')
    ax.plot(range(epochs), val_loss, label='Validation Loss')
    ax.plot(range(epochs), test_loss, label='Test Loss')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.savefig('{}_loss_{}.png'.format(data, suff),
                transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
