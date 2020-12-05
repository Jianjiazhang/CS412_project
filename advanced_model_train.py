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
from sklearn.metrics import mean_squared_error,r2_score

from opts import get_opts
ARGS = get_opts()

def main():
    file = ARGS.data_path + ARGS.saving_file
    with open(file) as f:
        data = json.load(f)

    subdata = data['1']
    print(len(np.array(subdata['time'][:])))
    firstNSequence = ARGS.data_size
    firstDColumn = 2

    firstNColData = np.zeros((firstNSequence, firstDColumn))

    startDate = datetime.datetime(2004, 2, 28, 0, 58, 15)
    startDateInSeconds = 0

    firstNColData[:, 0] = np.array(subdata['time'][:firstNSequence]) + startDateInSeconds
    firstNColData[:, 1] = np.array(subdata[ARGS.column][:firstNSequence])


    # convert to Pandas dataframe and set as datetimeindex
    #df = pd.DataFrame(firstNColData, columns = ['time', 'voltage', 'temperature', 'humidity', 'light'])
    df = pd.DataFrame(firstNColData, columns = ['time', ARGS.column])
    df['time'] = pd.to_datetime(df["time"], dayfirst = '', unit = 's')
    df = df.set_index('time')

    # # plot by day
    # df.temperature.resample('S').sum().plot(title = 'temperature sum by day')
    # plt.show()



    # split the data to train and validation sets
    values = df.values
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
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
    MIN_TW = 1
    TW = ARGS.tw
    all_reframed = [series_to_supervised(scaled, i, 1) for i in range(MIN_TW, TW + 1)]
    #reframed = series_to_supervised(scaled, TW, 1)

    # add the new 4 cols of (t-1) and one col of t which is voltage

    #reframed.drop(reframed.columns[[2]], axis=1, inplace=True)
    train_xs = []
    train_ys = []
    test_xs = []
    test_y = None
    for reframed in all_reframed:
        values = reframed.values

        # use first k to predict the rest days
        k = int(0.2 * firstNSequence)
        train = values[:(len(values) - k), :]
        test = values[(len(values) - k):, :]

        # split into inputs and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        # reshpape to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        train_xs.append(train_X)
        train_ys.append(train_y)
        test_xs.append(test_X)

    # fit date: use 100 neurons for the first layer
    models = []

    for i in range(MIN_TW, TW + 1):
        print('model {}'.format(i))
        train_X = train_xs[i - MIN_TW]
        train_y = train_ys[i - MIN_TW]
        test_X = test_xs[i - MIN_TW]
        numberOfNeurons = i * 2
        model = Sequential()
        model.add(LSTM(numberOfNeurons, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(train_X, train_y, epochs=ARGS.epochs, batch_size=ARGS.batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        models.append(model)

    # get new train X datasets
    combined_train_xs = []
    combined_data_size = train_xs[-1].shape[0]
    for i, model in enumerate(models):
        train_x = train_xs[i]
        y = model.predict(train_x)
        combined_train_xs.append(y[-combined_data_size:, 0])
    combined_train_xs = np.array(combined_train_xs).T
    combined_train_y = train_ys[-1]

    # train combine model
    combine_model = Sequential()
    combine_model.add(Dense(20, input_dim=TW-MIN_TW+1, kernel_initializer='normal', activation='relu'))
    combine_model.add(Dense(1, kernel_initializer='normal'))
    combine_model.compile(loss='mean_squared_error', optimizer='adam')
    combine_model.fit(combined_train_xs, combined_train_y, epochs=50, batch_size=4, verbose=2, shuffle=False)

    # make a prediction
    result = np.zeros(len(test_y))
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))

    inv_y = scaler.inverse_transform(test_y)
    inv_y = inv_y[:, 0]
    all_predictions = []
    for i, model in enumerate(models):
        test_X = test_xs[i]
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], -1))
        # invert scaling for forecast
        inv_yhat = np.concatenate((yhat, test_X), axis=1)
        inv_yhat_no_transform = inv_yhat[:, 0]
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        all_predictions.append(inv_yhat_no_transform)
        result +=  (inv_yhat / TW)
        # calculate RMSE
        mse = mean_squared_error(inv_y, inv_yhat)
        rmse = np.sqrt(mse)
        print('Test MSE: %.6f' % mse)

    all_predictions = np.array(all_predictions).T
    combined_result = combine_model.predict(all_predictions)
    combined_result = scaler.inverse_transform(combined_result)
    mse = mean_squared_error(inv_y, combined_result)
    print('Combined Test MSE: %.6f' % mse)

    # plot the diff: just show first m
    m = firstNSequence - TW
    aa=[x for x in range(len(inv_y))]
    plt.plot(aa, inv_y, label="actual")
    plt.plot(aa, combined_result, 'r', label="prediction")
    plt.ylabel(ARGS.column, size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.savefig(ARGS.column + '_ensemble.png', dpi=300)
    #plt.show()

if __name__ == '__main__':
    main()