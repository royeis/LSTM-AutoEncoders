import torch
from torch import nn


class LSTM_AutoEncoder(nn.Module):
    def __init__(
            self,
            input_size,
            enc_hidden_size,
            dec_hidden_size,
            enc_n_layers,
            dec_n_layers,
            activation
    ):
        super(LSTM_AutoEncoder, self).__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=enc_hidden_size,
            num_layers=enc_n_layers,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=enc_hidden_size,
            hidden_size=dec_hidden_size,
            num_layers=dec_n_layers,
            batch_first=True
        )

        self.linear = nn.Linear(dec_hidden_size, input_size)

        # we will flow, maximum it will throw.
        self.act = activation()

    def forward(self, batch):
        _, (z, _) = self.encoder_lstm(batch)
        z = z.reshape(batch.shape[0], 1, -1).expand(-1, batch.shape[1], -1)
        h_s, _ = self.decoder_lstm(z)
        x_rec = self.act(self.linear(h_s))
        return x_rec


