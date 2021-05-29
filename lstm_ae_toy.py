import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import trange

from Model import LSTM_AutoEncoder
from generate_toy_dataset import create_toy_data
from Utils import prepare_tensor_dataset, set_all_seeds, plot_signals_with_rec



if __name__ == '__main__':
    set_all_seeds(2021)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device used: {device}')

    do_train = 1
    do_val = 1
    do_test = 0

    x_train, x_validate, x_test = create_toy_data()
    # plot_signals(x_train[:3])
    train_dataset = prepare_tensor_dataset(x_train)
    validate_dataset = prepare_tensor_dataset(x_validate)
    test_dataset = prepare_tensor_dataset(x_test)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = LSTM_AutoEncoder(input_size=1,
                             enc_hidden_size=512,
                             dec_hidden_size=512,
                             enc_n_layers=1,
                             dec_n_layers=1,
                             activation=nn.Sigmoid
                             ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='sum')

    if do_train:
        t = trange(400)
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

            # validation after each training epoch
            if do_val:
                valid_losses = []
                model.eval()
                with torch.no_grad():
                    for val_batch in validate_dataloader:
                        x = val_batch[0].to(device)
                        x_rec = model(x)
                        loss = criterion(x, x_rec)
                        valid_losses.append(loss.item())
                if epoch % 10 == 0:
                    plot_signals_with_rec(x[:3].detach().cpu(), x_rec[:3].detach().cpu())

                t.set_description('epoch {} train loss {:.2f} valid loss{:.2f} '
                                  .format(epoch, np.mean(losses), np.mean(valid_losses)))
        torch.save(model.state_dict(), 'toy_data_model_weights.pth')

    if do_test:
        model.load_state_dict(torch.load('toy_data_model_weights.pth'))
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_dataloader:
                x = batch[0].to(device)
                x_rec = model(x)
                loss = criterion(x, x_rec)
                test_losses.append(loss.item())
        plot_signals_with_rec(x[:3].detach().cpu(), x_rec[:3].detach().cpu())
        print('test loss: {:.2f}'.format(np.mean(test_losses)))