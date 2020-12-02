import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import statsmodels.api as sm

class LSTM1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_layer_size = args.hidden_layer_size

        self.lstm = nn.LSTM(args.input_size, args.hidden_layer_size,num_layers=args.num_layers)
        self.dropout = nn.Dropout(args.dp)

        self.linear = nn.Linear(args.hidden_layer_size, args.output_size)

        

    def forward(self, input_seq):

        Tsteps = input_seq.shape[1]
        self.hidden_cell = (torch.zeros(self.args.num_layers,Tsteps,self.hidden_layer_size),
                            torch.zeros(self.args.num_layers,Tsteps,self.hidden_layer_size))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,Tsteps, -1), self.hidden_cell)
        predictions = self.linear(lstm_out)
        # print(predictions.shape)
        predictions = predictions[:,-1,:]
        predictions = torch.squeeze(predictions)
        # print(predictions.shape)
        # exit()
        return predictions