import numpy as np
import json
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from opts import get_opts
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error,r2_score
file = 'data/data_knn_interp.json'
with open(file) as f:
    data = json.load(f)
args = get_opts()
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

### (12000, 1, 4) ###
# print(train_X.shape)
print('>>>>> Feature preprocess is done <<<<<')
from three2one_model import LSTM1



class create_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data,labels,args):

        self.data = data
        self.labels = labels
        self.args = args
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        re = self.data[idx]
        re_label = self.labels[idx]
        return np.array(re),np.array(re_label)

def create_sequences(data,label,args):
    xs = []
    ys = []
    for i in range(len(data)-args.seq_length-1):
        x = data[i:(i+args.seq_length)]
        y = label[i+args.seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

train_data,train_labels = create_sequences(train_X, train_y,args)
test_data,test_labels = create_sequences(test_X, test_y,args)
train_data = np.squeeze(train_data,axis=2)
test_data = np.squeeze(test_data,axis=2)

train_sets = create_dataset(train_data, train_labels,args)
test_sets = create_dataset(test_data, test_labels,args)

train_loader = torch.utils.data.DataLoader(train_sets, 
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            num_workers=2,
                                            drop_last=True)

test_loader = torch.utils.data.DataLoader(test_sets, 
                                            batch_size=args.batch_size, 
                                            shuffle=False, 
                                            num_workers=2,
                                            drop_last=True)




def train(train,test,args):
    model1 = LSTM1(args)
    # model2 = LSTM2(args)
    loss_function = nn.MSELoss()

    ### need more optimizers ###
    '''
        1. For each optimizer, we can try different parameters
        2. For each model, we can try different optimizers
    '''
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)


    if args.debug:
        args.epochs = 1
    for i in range(args.epochs):
        for seq, labels in train:

            optimizer1.zero_grad()
            # optimizer2.zero_grad()

            model1 = model1.float()
            # model2 = model2.float()

            seq = seq.numpy()
            seq = torch.from_numpy(seq).float()

            labels = labels.numpy()
            labels = torch.from_numpy(labels).float()

            y_pred1 = model1(seq)

            # y_pred2 = model2(seq)
            single_loss1 = loss_function(y_pred1, labels)
            # single_loss2 = loss_function(y_pred2, labels)

            # loss = (single_loss1 + single_loss2)/2
            # loss.backward()
            '''
                loss = (single_loss1 + single_loss2)/N
                loss.backward()
                Using total aveg loss to train the model, model A's traing will bring influence to model B
            '''
            # single_loss1 = single_loss1.float()
            single_loss1.backward()

            # single_loss2.backward()
            '''
                Using individual loss to train the model, model A's traing will not bring influence to model B
            '''
            optimizer1.step()
            # optimizer2.step()

        print(f'epoch: {i:3} loss: {single_loss1.item():10.8f}')

    ### Place to ensemble the results ###
    actual_predictions = evaluate(model1,args,test)
    ### compute MSE ###
    # print(f"MSE：{mean_squared_error(actual_predictions, data[-args.num_fut:])}")

def evaluate(model,args,test):
    model.eval()
    pre = []
    true = []
    for seq, labels in test:
        model = model.float()
            # model2 = model2.float()

        seq = seq.numpy()
        seq = torch.from_numpy(seq).float()

        labels = labels.numpy()
        labels = torch.from_numpy(labels).float()

        y_pred1 = model(seq)


        pre.append(y_pred1.detach().numpy())
        true.append(labels.detach().numpy())

    actual_predictions = np.array(pre).reshape(-1)
    true = np.array(true).reshape(-1)
    ### compute MSE ###
    print(f"MSE：{mean_squared_error(actual_predictions, true)}")
    return actual_predictions


train(train_loader,test_loader,args)


