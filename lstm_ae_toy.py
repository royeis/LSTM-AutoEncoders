import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
import argparse

from Model import LSTM_AutoEncoder
from generate_toy_dataset import create_toy_data
from Utils import prepare_tensor_dataset, set_all_seeds, plot_signals_with_rec, plot_signals, plot_loss

# hyperparameters
parser = argparse.ArgumentParser(description="Toy Dataset Task")
parser.add_argument("--epochs", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--clip", type=float, default=1)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_size", type=int, default=25)
parser.add_argument("--seed", type=int, default=2021)

args = parser.parse_args()

if __name__ == '__main__':
    set_all_seeds(args.seed)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device used: {device}')

    suff = 'lr={:.5f}_bs={}_hs={}_clip={:.2f}'.format(args.lr, args.batch_size, args.hidden_size,
                                                      args.clip)
    do_train = 1
    do_val = 1
    do_test = 1
    do_plot_examples = 0

    x_train, x_validate, x_test = create_toy_data()
    # plot_signals(x_train[:3])
    train_dataset = prepare_tensor_dataset(x_train)
    validate_dataset = prepare_tensor_dataset(x_validate)
    test_dataset = prepare_tensor_dataset(x_test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = LSTM_AutoEncoder(input_size=1,
                             enc_hidden_size=args.hidden_size,
                             dec_hidden_size=args.hidden_size,
                             enc_n_layers=args.num_layers,
                             dec_n_layers=args.num_layers,
                             activation=nn.Sigmoid
                             ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='sum')

    if do_plot_examples:
        for val_batch in validate_dataloader:
            x = val_batch[0].to(device)
            plot_signals(x[:3].detach().cpu())
            break

    if do_train:
        # model.load_state_dict(torch.load('toy_data_model_weights_{}.pth'.format(suff)))
        t = trange(args.epochs)
        train_losses, val_losses, test_losses = [], [], []

        for epoch in t:
            model.train()
            epoch_losses = []
            for batch in train_dataloader:
                x = batch[0].to(device)
                x_rec, _ = model(x)
                loss = criterion(x, x_rec)
                epoch_losses.append(loss.item())
                x.detach()
                x_rec.detach()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()

            train_losses.append(np.mean(epoch_losses))
            # validation after each training epoch
            valid_losses = []
            if do_val:
                model.eval()
                with torch.no_grad():
                    for val_batch in validate_dataloader:
                        x = val_batch[0].to(device)
                        x_rec, _ = model(x)
                        loss = criterion(x, x_rec)
                        valid_losses.append(loss.item())

                val_losses.append(np.mean(valid_losses))

            t_losses = []
            if do_test:
                # model.load_state_dict(torch.load('toy_data_model_weights_{}.pth'.format(suff)))
                model.eval()
                with torch.no_grad():
                    for batch in test_dataloader:
                        x = batch[0].to(device)
                        x_rec, _ = model(x)
                        loss = criterion(x, x_rec)
                        t_losses.append(loss.item())
                # plot_signals_with_rec(x[:3].detach().cpu(), x_rec[:3].detach().cpu(), args)
                # print('test loss: {:.2f}'.format(np.mean(test_losses)))

                test_losses.append(np.mean(t_losses))

            if do_val == 0 and do_test == 0:
                t.set_description('epoch {} train loss {:.2f} '
                                  .format(epoch, np.mean(epoch_losses)))

            t.set_description('epoch {} train loss {:.2f} valid loss {:.2f} test loss {:.2f}'
                              .format(epoch, np.mean(epoch_losses), np.mean(valid_losses), np.mean(t_losses)))
            torch.save(model.state_dict(), 'toy_data_model_weights_{}.pth'.format(suff))
        plot_loss(train_losses, val_losses, test_losses, args.epochs, suff, parser.description)
