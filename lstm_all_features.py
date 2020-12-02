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
file = 'data/data_knn_interp.json'
with open(file) as f:
    data = json.load(f)

subdata = data['1']
print(len(np.array(subdata['time'][:])))
firstNSequence = 15000
firstDColumn = 5

firstNColData = np.zeros((firstNSequence, firstDColumn))

startDate = datetime.datetime(2004, 2, 28, 0, 58, 15)
startDateInSeconds = int(startDate.strftime('%s'))

firstNColData[:, 0] = np.array(subdata['time'][:firstNSequence]) + startDateInSeconds
firstNColData[:, 1] = np.array(subdata['voltage'][:firstNSequence])
firstNColData[:, 2] = np.array(subdata['temperature'][:firstNSequence])
firstNColData[:, 3] = np.array(subdata['humidity'][:firstNSequence])
firstNColData[:, 4] = np.array(subdata['light'][:firstNSequence])


# convert to Pandas dataframe and set as datetimeindex
df = pd.DataFrame(firstNColData, columns = ['time', 'temperature', 'humidity', 'light', 'voltage'])
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
reframed = series_to_supervised(scaled, 1, 1)

# add the new 4 cols of (t-1) and one col of t which is voltage

reframed.drop(reframed.columns[[5,6,7]], axis=1, inplace=True)
values = reframed.values

# use first k to predict the rest days
k = int(0.8 * firstNSequence)
train = values[:k, :]
test = values[k:, :]

# split into inputs and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshpape to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# fit date: use 100 neurons for the first layer
numberOfNeurons = 100
model = Sequential()
model.add(LSTM(numberOfNeurons, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot the loss function
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], -1))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -3:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -3:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.6f' % rmse)


# plot the diff: just show first m
m = firstNSequence - 1
aa=[x for x in range(m)]
plt.plot(aa, np.concatenate([firstNColData[:k, 1], inv_y]) , marker='.', label="actual")
plt.plot(aa,  np.concatenate([firstNColData[:k, 1], inv_yhat]) , 'r', label="prediction")
plt.ylabel('Voltage', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
