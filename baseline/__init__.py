import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import json
from sklearn.preprocessing import MinMaxScaler
from model import LSTM

TRAIN_DATA_SIZE = 0.6
VALIDATION_DATA_SIZE = 0.2
TRIAN_WINDOW = 60
EPOCHS = 50

def min_max_scale(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 4))
    print(train_data_normalized[:5])
    return train_data_normalized

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        data = json.load(f)
        f.close()
    data = data['1']    # only use the node 1
    all_data = np.array([data['temperature'], data['humidity'], data['light'], data['voltage']]).T

    size = len(all_data) if sys.argv[2] == -1 else int(sys.argv[2])
    all_data = all_data[:size]
    train_data = all_data[:int(len(all_data) * TRAIN_DATA_SIZE)]
    valid_data = all_data[int(len(all_data) * TRAIN_DATA_SIZE):int(len(all_data) * TRAIN_DATA_SIZE + len(all_data) * VALIDATION_DATA_SIZE)]
    test_data = all_data[int(len(all_data) * TRAIN_DATA_SIZE + len(all_data) * VALIDATION_DATA_SIZE):]
    print('train data len:', len(train_data))
    print('valid data len:', len(valid_data))
    print('test data len:', len(test_data))
    print('first 5 data:\n', all_data[:5])

    train_data_normalized = train_data.reshape(-1, 4)
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 4))

    print('scaled first 5 data:\n', train_data[:5])

    train_data_normalized = torch.FloatTensor(train_data_normalized).cuda()
    
    train_inout_seq = create_inout_sequences(train_data_normalized, TRIAN_WINDOW)

    nn_model = LSTM()
    nn_model.cuda()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    print('baseline model:', nn_model)

    # train model
    for i in range(EPOCHS):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            nn_model.hidden_cell = (torch.zeros(1, 1, nn_model.hidden_layer_size).cuda(),
                            torch.zeros(1, 1, nn_model.hidden_layer_size).cuda())

            y_pred = nn_model(seq)
            single_loss = loss_function(y_pred, labels.reshape(4))
            single_loss.backward()
            optimizer.step()
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'final epoch: {i:3} loss: {single_loss.item():10.10f}')

    # inference
    fut_pred = len(valid_data)
    test_inputs = train_data_normalized[-TRIAN_WINDOW:].tolist()

    nn_model.eval()
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-TRIAN_WINDOW:]).cuda()
        with torch.no_grad():
            nn_model.hidden = (torch.zeros(1, 1, nn_model.hidden_layer_size).cuda(),
                            torch.zeros(1, 1, nn_model.hidden_layer_size).cuda())
            predicted = list(nn_model(seq))
            test_inputs.append(predicted)

    #actual_predictions = scaler.inverse_transform(np.array(test_inputs[TRIAN_WINDOW:]))
    actual_predictions = np.array(test_inputs[TRIAN_WINDOW:])
    mse = ((actual_predictions - valid_data)**2).mean(0)
    mse = [tensor.item() for tensor in mse]
    print('mse for each column:', mse)
    print('sum of mse', sum(mse))