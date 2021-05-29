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
            activation,
            n_categories=None
    ):
        super(LSTM_AutoEncoder, self).__init__()

        self.n_categories = n_categories

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
        if self.n_categories:
            self.linear_classifier = nn.Linear(dec_hidden_size, self.n_categories)

        self.act = activation()

    def forward(self, batch):
        _, (z, _) = self.encoder_lstm(batch)
        print(z.shape)
        z = z.reshape(batch.shape[0], 1, -1).expand(-1, batch.shape[1], -1)
        h_s, (h_t, _) = self.decoder_lstm(z)
        x_rec = self.act(self.linear(h_s))
        if self.n_categories:
            c = self.linear_classifier(h_t.squeeze())
            return x_rec, c
        return x_rec, None


