import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=1, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size).cuda()

        self.linear = nn.Linear(hidden_layer_size, output_size).cuda()

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).cuda(),
                            torch.zeros(1, 1, self.hidden_layer_size).cuda())
        self.dropout = nn.Dropout(0.5)

    def reset_hidden_state(self):
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).cuda(),
                            torch.zeros(1, 1, self.hidden_layer_size).cuda())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
