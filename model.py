import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import statsmodels.api as sm


class LSTM1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_layer_size = args.hidden_layer_size

        self.lstm = nn.LSTM(args.input_size, args.hidden_layer_size).cuda()

        self.linear = nn.Linear(args.hidden_layer_size, args.output_size).cuda()

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).cuda(),
                            torch.zeros(1, 1, self.hidden_layer_size).cuda())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class LSTM2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_layer_size = args.hidden_layer_size//2

        self.lstm = nn.LSTM(args.input_size, args.hidden_layer_size//2).cuda()

        self.linear = nn.Linear(args.hidden_layer_size//2, args.output_size).cuda()

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size//2).cuda(),
                            torch.zeros(1, 1, self.hidden_layer_size//2).cuda())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class SArimax(object):
    """docstring for SArimax"""

    def __init__(self, data):
        super(SArimax, self).__init__()
        self.data = data

    def build_model(self):
        return sm.tsa.statespace.SARIMAX(self.data)

    def predict(self):
        model = self.build_model()
        result = model.fit(self.data)
        re = result.predict()
        return re
