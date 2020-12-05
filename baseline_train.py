import tensorflow as tf
import numpy as np
import json
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# ipython --matplotlib

# for deep learning
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error, r2_score

from opts import get_opts
ARGS = get_opts()


# frame as supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def preprocess(data):
    subdata = data['1']
    print(len(np.array(subdata['time'][:])))

    truncated_data = np.zeros((ARGS.data_size, 2))

    start_date_in_seconds = 0

    truncated_data[:, 0] = np.array(
        subdata['time'][:ARGS.data_size]) + start_date_in_seconds
    truncated_data[:, 1] = np.array(subdata[ARGS.column][:ARGS.data_size])

    # convert to Pandas dataframe and set as datetimeindex
    df = pd.DataFrame(truncated_data, columns=['time', 'temperature'])
    df['time'] = pd.to_datetime(df["time"], dayfirst='', unit='s')
    df = df.set_index('time')

    # split the data to train and validation sets
    values = df.values
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    reframed = series_to_supervised(scaled, ARGS.tw, 1)

    # add the new 4 cols of (t-1) and one col of t which is voltage

    values = reframed.values

    # use first k to predict the rest days
    k = int(0.2 * ARGS.data_size)
    train = values[:(len(values) - k), :]
    test = values[(len(values) - k):, :]
    # split into inputs and outputs
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    # reshpape to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print('train data length:', len(train_x))
    print('test data length:', len(test_x))
    return train_x, train_y, test_x, test_y, scaler


def train(train_x, train_y, test_x, test_y):
    # fit date: use 100 neurons for the first layer
    num_neurons = 20
    model1 = Sequential()
    model1.add(LSTM(num_neurons, input_shape=(
        train_x.shape[1], train_x.shape[2])))
    model1.add(Dropout(0.5))
    model1.add(Dense(1))
    model1.compile(loss='mean_squared_error', optimizer='adam')
    model1.fit(train_x, train_y, epochs=ARGS.epochs, batch_size=32, validation_data=(
        test_x, test_y), verbose=2, shuffle=False)

    return model1


def predict(test_x, test_y, model1, scaler):
    # make a prediction
    yhat1 = model1.predict(test_x)
    yhat2 = model2.predict(test_x)

    test_x = test_x.reshape((test_x.shape[0], -1))
    # invert scaling for forecast
    inv_yhat1 = np.concatenate((yhat1, test_x), axis=1)
    inv_yhat1 = scaler.inverse_transform(inv_yhat1)
    inv_yhat1 = inv_yhat1[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_x), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    mse1 = mean_squared_error(inv_y, inv_yhat1)
    rmse1 = np.sqrt(mse1)
    print('Test MSE of the model1: %.6f' % mse1)


    # plot the diff: just show first m
    aa=[x for x in range(len(inv_y))]
    plt.plot(aa, inv_y, label="actual")
    plt.plot(aa, inv_yhat1, 'r', label="prediction")
    plt.ylabel(ARGS.column, size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.savefig(ARGS.column + '_baseline.png', dpi=300)
    #plt.show()


if __name__ == '__main__':
    tf.get_logger().setLevel('INFO')
    file = ARGS.data_path + ARGS.saving_file
    with open(file) as f:
        data = json.load(f)

    train_x, train_y, test_x, test_y, scaler = preprocess(data)
    model1 = train(train_x, train_y, test_x, test_y)
    predict(test_x, test_y, model1, scaler)
