import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from tqdm import trange
import matplotlib.pyplot as plt

from Model import LSTM_AutoEncoder


def set_all_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    classify = 1

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if device == torch.device('cuda:0'):
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # inp_size = dataset1.train_data[0].shape[0]
    inp_size = 28

    model = LSTM_AutoEncoder(input_size=inp_size,
                             enc_hidden_size=128,
                             dec_hidden_size=128,
                             enc_n_layers=1,
                             dec_n_layers=1,
                             activation=nn.Sigmoid,
                             n_categories=10
                             ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction='sum')
    criterion_classify = nn.CrossEntropyLoss(reduction='mean')

    if do_train:
        t = trange(250)
        x, x_rec = None, None
        for epoch in t:
            model.train()
            losses = []
            for batch in train_loader:
                x = batch[0].squeeze().to(device)
                y = batch[1].to(device)
                # x = x.view(x.shape[0], -1, 28)

                x_rec, c = model(x)

                loss = criterion(x, x_rec)

                if classify:
                    loss += criterion_classify(c, y)
                losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

                x.detach()
                x_rec.detach()
                c.detach()

                t.set_description('epoch {} train_loss {:.2f} '
                                  .format(epoch, np.mean(losses)))

        torch.save(model.state_dict(), 'mnist_data_model_weights_one_clip5_hs64.pth')

    if do_test:
        torch.load('mnist_data_model_weights_one_clip5_hs64.pth')
        model.eval()
        with torch.no_grad():
            for batch in train_loader:

                x = batch[0].squeeze().to(device)
                y = batch[1]
                x_rec, c = model(x)

                c_t = np.argmax(F.softmax(c).detach().cpu().numpy(), axis=1)
                y = y.detach().cpu().numpy()
                print(c_t)
                print(y)
                # x_rec = x_rec.view(x.shape[0], -1, 28)
                # for ii in range(4):
                #     fig = plt.figure
                #     plt.imshow(x[ii].detach().cpu().numpy(), cmap='gray')
                #     plt.show()
                #     plt.imshow(x_rec[ii].detach().cpu().numpy(), cmap='gray')
                #     plt.show()

                break
