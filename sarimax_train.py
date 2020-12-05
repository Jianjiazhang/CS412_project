import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from opts import get_opts
ARGS = get_opts()

def main():
    file = ARGS.data_path + ARGS.saving_file
    with open(file, 'r') as f:
        data = json.load(f)
        f.close()

    series = pd.DataFrame(np.array(data['1'][ARGS.column][:ARGS.data_size]), columns=[ARGS.column])
    X = series.values
    size = int(len(X) * 0.8)
    train, test = X[0:size], X[size:len(X)]

    model = sm.tsa.statespace.SARIMAX(train, order=(1, 1, 0), seasonal_order=(1, 1, 1, 10))
    model_fit = model.fit()
    yhat = model_fit.forecast(len(test))
    error = mean_squared_error(test, yhat)
    print('Test MSE: %.6f' % error)

    # plot
    predictions = np.concatenate((train, yhat), axis=None)
    pyplot.plot(yhat, color='red', label='predicted')
    pyplot.plot(test, label='actual')
    pyplot.ylabel(ARGS.column, size=15)
    pyplot.xlabel('Time step', size=15)
    pyplot.legend(fontsize=15)
    pyplot.savefig(ARGS.column + '_SARIMAX.png', dpi=300)
    #pyplot.show()

if __name__ == '__main__':
    main()