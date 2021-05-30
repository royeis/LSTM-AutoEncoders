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

    def forward(self, batch, seq_len=None):
        print('\nbatch')
        print('#'*20)
        print(batch)
        print('#' * 20)
        outputs, (z, _) = self.encoder_lstm(batch)

        if seq_len is not None:
            z = self.extract_relevant_states(outputs, seq_len)

        z = z.reshape(batch.shape[0], 1, -1).expand(-1, batch.shape[1], -1)
        print('#'*20)
        print(z.shape)
        h_s, (h_t, _) = self.decoder_lstm(z)

        # if seq_len is not None:
        #     h_t = self.extract_relevant_states(h_s, seq_len)

        x_rec = self.act(self.linear(h_s))
        if self.n_categories:
            c = self.linear_classifier(h_t.squeeze())
            return x_rec, c
        return x_rec, None

    def extract_relevant_states(self, outputs, seq_len):
        out = []
        print('outputs')
        print(outputs)
        print(outputs.shape)
        print(seq_len)
        # create loop to collect only relevant outputs for the batch
        for i, idx in enumerate(seq_len):
            out.append(outputs[i][idx - 1])
        z = torch.stack(out)
        print(z)
        print(z.shape)
        return z


