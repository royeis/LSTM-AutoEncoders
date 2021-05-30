import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import trange


from Model import LSTM_AutoEncoder
from Utils import plot_signals_with_rec


def set_all_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_stock_data(stock_symbol):
    data = stocks[stocks.index == stock_symbol]
    data = data.reset_index(drop=True)
    data = data['high'].values
    return data, len(data)


def normalize_stock_data(stock_data):
    stock_data = (stock_data - np.mean(stock_data)) / np.std(stock_data)
    return stock_data


if __name__ == '__main__':

    # hyperparameters
    parser = argparse.ArgumentParser(description="MNIST Task")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2021)

    args = parser.parse_args()

    set_all_seed(args.seed)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    do_train = 1
    do_test = 1
    do_val = 0

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if device == torch.device('cuda:0'):
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    stocks = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
    stock_symobls = stocks['symbol'].unique().tolist()
    test_symbols = set(random.sample(stock_symobls, int(0.2 * len(stock_symobls))))

    stocks = stocks.set_index('symbol', drop=True)
    train_df = stocks.drop(test_symbols, axis=0)
    test_df = stocks.drop(stocks.index.difference(test_symbols), axis=0)

    train_symbols = train_df.index.unique().tolist()

    train_tensors = []
    train_seq_lens = []
    for sym in train_symbols:
        stock_data, stock_data_len = prepare_stock_data(sym)
        stock_data = normalize_stock_data(stock_data)
        stock_tensor = torch.Tensor(stock_data)
        train_seq_lens.append(stock_data_len)
        train_tensors.append(stock_tensor)
    X = pad_sequence(train_tensors).T.unsqueeze(-1)
    y = torch.Tensor(train_seq_lens)
    train_dataset = TensorDataset(X, y)

    test_seq_lens = []
    test_tensors = []
    for sym in test_symbols:
        stock_data, stock_data_len = prepare_stock_data(sym)
        stock_data = normalize_stock_data(stock_data)
        test_seq_lens.append(stock_data_len)
        stock_tensor = torch.Tensor(stock_data)
        test_tensors.append(stock_tensor)
    X = pad_sequence(test_tensors).T.unsqueeze(-1)
    y = torch.Tensor(test_seq_lens)
    test_dataset = TensorDataset(X, y)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    inp_size = 1

    model = LSTM_AutoEncoder(input_size=inp_size,
                             enc_hidden_size=256,
                             dec_hidden_size=256,
                             enc_n_layers=1,
                             dec_n_layers=1,
                             activation=nn.Sigmoid,
                             ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if do_train:
        t = trange(150)
        for epoch in t:
            model.train()
            losses = []
            for batch in train_loader:
                x = batch[0].to(device)
                lens = batch[1].squeeze().long()

                x_rec, _ = model(x, lens)
                loss = criterion(x, x_rec)
                print('\nloss')
                print(loss.item())

                losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                x.detach()
                x_rec.detach()

                t.set_description('epoch {} train_loss {:.2f} '
                                  .format(epoch, np.mean(losses)))

        torch.save(model.state_dict(), 'sp_model_hs128.pth')

    if do_test:
        model.load_state_dict(torch.load('sp_model_hs128.pth'))
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                lens = batch[1].squeeze().long()

                x_rec, _ = model(x)
                loss = criterion(x, x_rec)
                test_losses.append(loss.item())
        plot_signals_with_rec(x[:3].detach().cpu(), x_rec[:3].detach().cpu())
        print('test loss: {:.2f}'.format(np.mean(test_losses)))




