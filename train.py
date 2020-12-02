import numpy as np
import json
from opts import get_opts
from model import LSTM1, LSTM2, SArimax
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing


def create_inout_sequences(input_data, args):
    inout_seq = []
    L = len(input_data)
    for i in range(L-args.tw):
        train_seq = input_data[i:i+args.tw]
        train_label = input_data[i+args.tw:i+args.tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def train(data, args):
    model1 = LSTM1(args)
    model2 = LSTM2(args)
    loss_function = nn.MSELoss()

    ### need more optimizers ###
    '''
        1. For each optimizer, we can try different parameters
        2. For each model, we can try different optimizers
    '''
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)

    train_data = torch.FloatTensor(data).view(-1)

    ### sarimax model ###
    normalized = preprocessing.normalize(data.reshape(-1, 1))
    sarimax = SArimax(normalized)
    re = sarimax.predict()
    #####################

    train_inout_seq = create_inout_sequences(train_data, args)
    if args.debug:
        args.epochs = 1
    for i in range(args.epochs):
        for seq, labels in train_inout_seq:
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            model1.hidden_cell = (torch.zeros(1, 1, model1.hidden_layer_size),
                                  torch.zeros(1, 1, model1.hidden_layer_size))

            y_pred1 = model1(seq)
            y_pred2 = model2(seq)

            single_loss1 = loss_function(y_pred1, labels)
            single_loss2 = loss_function(y_pred2, labels)

            # loss = (single_loss1 + single_loss2)/2
            # loss.backward()
            '''
                loss = (single_loss1 + single_loss2)/N
                loss.backward()
                Using total aveg loss to train the model, model A's traing will bring influence to model B
            '''
            single_loss1.backward()
            single_loss2.backward()
            '''
                Using individual loss to train the model, model A's traing will not bring influence to model B
            '''
            optimizer1.step()
            optimizer2.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    ### Place to ensemble the results ###
    actual_predictions = evaluate(model1, args, data)
    ### compute MSE ###
    print(
        f"MSE：{mean_squared_error(actual_predictions, data[-args.num_fut:])}")


def evaluate(model, args, data):
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
    # print(f"MSE：{mean_squared_error(actual_predictions, data[-args.num_fut:])}")
    return actual_predictions


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
