import numpy as np
import json
from opts import get_opts
from model import LSTM
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score


def create_inout_sequences(input_data, args):
    inout_seq = []
    L = len(input_data)
    for i in range(L-args.tw):
        train_seq = input_data[i:i+args.tw]
        train_label = input_data[i+args.tw:i+args.tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def train(data, args):
    model = LSTM(args)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_data = torch.FloatTensor(data).view(-1)
    train_inout_seq = create_inout_sequences(train_data, args)
    if args.debug:
        args.epochs = 1
    for i in range(args.epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    ### model evaluation ###
    model.eval()
    test_inputs = data[-args.num_fut:].tolist()
    for i in range(args.num_fut):
        seq = torch.FloatTensor(test_inputs[-args.num_fut:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())
    actual_predictions = np.array(test_inputs[args.num_fut:]).reshape(-1)
    ### compute MSE ###
    print(
        f"MSE：{mean_squared_error(actual_predictions, data[-args.num_fut:])}")


def main(args):
    file = args.data_path+args.saving_file
    with open(file) as f:
        data = json.load(f)

    '''
        Attention: 
        1. Only using node 1
        2. Only using first 1000 temperature data 
    '''
    subdata = data['1']
    input_seq = np.array(subdata['temperature'][:1000])

    print('>>>>> Input sequence has been created <<<<<')

    train(input_seq, args)


if __name__ == '__main__':
    args = get_opts()
    main(args)
