import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_layer_size = args.hidden_layer_size

        self.lstm = nn.LSTM(args.input_size, args.hidden_layer_size)

        self.linear = nn.Linear(args.hidden_layer_size, args.output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
