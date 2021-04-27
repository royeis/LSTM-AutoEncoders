import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt

from Model import LSTM_AutoEncoder
from generate_toy_dataset import create_toy_data
from Utils import prepare_tensor_dataset


def plot_signals(signals):
    signals = signals.squeeze()
    seq_len = len(signals[0])
    fig, axs = plt.subplots(len(signals))
    fig.suptitle('Random Generated Signals')
    for sig, ax in zip(signals, axs):
        ax.plot(range(seq_len), sig)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.show()


def plot_signals_with_rec(signals, signals_rec):
    signals, signals_rec = signals.squeeze(), signals_rec.squeeze()
    seq_len = len(signals[0])
    fig, axs = plt.subplots(len(signals))
    fig.suptitle('Random Generated Signals')
    for sig, sig_rec, ax in zip(signals, signals_rec, axs):
        ax.plot(range(seq_len), sig, label="Original Signal")
        ax.plot(range(seq_len), sig_rec, label="Reconstructed Signal")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()
    fig.show()


def set_all_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_all_seed(2021)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    do_train = 0
    do_test = 0
    do_val = 1

    x_train, x_validate, x_test = create_toy_data()
    # plot_signals(x_train[:3])
    train_dataset = prepare_tensor_dataset(x_train)
    validate_dataset = prepare_tensor_dataset(x_validate)
    test_dataset = prepare_tensor_dataset(x_test)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = LSTM_AutoEncoder(input_size=1,
                             enc_hidden_size=256,
                             dec_hidden_size=256,
                             enc_n_layers=1,
                             dec_n_layers=1,
                             activation=nn.Sigmoid
                             ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='sum')

    if do_train:
        t = trange(500)
        for epoch in t:
            model.train()
            losses = []
            for batch in train_dataloader:
                x = batch[0].to(device)
                x_rec = model(x)
                loss = criterion(x, x_rec)
                losses.append(loss.item())
                x.detach()
                x_rec.detach()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            valid_losses = []
            model.eval()
            with torch.no_grad():
                for val_batch in validate_dataloader:
                    x = val_batch[0].to(device)
                    x_rec = model(x)
                    loss = criterion(x, x_rec)
                    valid_losses.append(loss.item())

                t.set_description('epoch {} train_loss {:.2f} valid_loss{:.2f} '
                                  .format(epoch, np.mean(losses), np.mean(valid_losses)))
            torch.save(model.state_dict(), 'toy_data_model_weights.pth')

    if do_test:
        torch.load('toy_data_model_weights.pth')
        model.eval()
        for batch in test_dataloader:
            x = batch[0].to(device)
            x_rec = model(x)
            plot_signals_with_rec(x[:3].detach().cpu(), x_rec[:3].detach().cpu())

    if do_val:
        model.eval()
        with torch.no_grad():
            for val_batch in validate_dataloader:
                x = val_batch[0].to(device)
                x_rec = model(x)
                plot_signals_with_rec(x[:3].detach().cpu(), x_rec[:3].detach().cpu())