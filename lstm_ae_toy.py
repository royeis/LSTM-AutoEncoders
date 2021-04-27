import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Model import LSTM_AutoEncoder
from generate_toy_dataset import create_toy_data
from Utils import prepare_tensor_dataset


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    data = create_toy_data()
    dataset = prepare_tensor_dataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LSTM_AutoEncoder(input_size=1,
                             enc_hidden_size=1024,
                             dec_hidden_size=1024,
                             enc_n_layers=1,
                             dec_n_layers=1,
                             activation=nn.Sigmoid
                             )
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    criterion = nn.MSELoss(reduction='sum')

    cur_loss = 10000
    for epoch in range(100):
        generator = tqdm(enumerate(dataloader), total=len(dataloader))
        generator.set_description('epoch {} loss {:.2f}'.format(epoch, cur_loss))
        losses = []
        for i, batch in generator:
            x = batch[0]
            x_rec = model(x)
            loss = criterion(x, x_rec)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        cur_loss = np.mean(losses)

    print('done')
