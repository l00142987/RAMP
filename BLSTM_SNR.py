import torch
import torch.nn as nn

class SNR_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SNR_Model, self).__init__()

        self.input_linear = nn.Linear(input_dim, hidden_dim)

        self.blstm_layers = nn.ModuleList([
            nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input_linear(x))

        for blstm in self.blstm_layers:
            x, _ = blstm(x)

        x = self.output_linear(x[:, -1, :])
        return x