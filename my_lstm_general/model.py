import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, model_from_json
from keras.layers import Input, Dense, LSTM, Dropout, Flatten
from math import sqrt
import sys
import copy


def visualize_stock_prices(stock_prices1, stock_prices2):
    plt.plot(stock_prices1, color='red', label='Actual Stock Prices')
    plt.plot(stock_prices2, color='green', label='Predicted Stock Prices')
    plt.title('Stock Prices Historical Data')
    plt.xlabel('Time (Days)')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# create a differenced series
def difference(dataset, interval=1):
    diff = [None] * interval
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff


# invert differenced value
def inverse_difference(history, y_predicted, interval=1):
    return y_predicted + history[-interval]


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(df, shift, lag=1):
    columns = [df.shift(i) for i in range(1, lag + 1)][shift:]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    return df


# scale train and test data to [-1, 1]
def scale(train, test, timesteps, shift, len_dataset):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train.reshape((timesteps - shift + 1) * (len_dataset - timesteps - shift - 30), train.shape[2]))
    # transform train
    train_scaled = np.array([scaler.transform(exp) for exp in train])
    # transform test
    test_scaled = np.array([scaler.transform(exp) for exp in test])
    return scaler, train_scaled, test_scaled


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons, timesteps):
    X, y = train[:, 0:-1], np.array([y[-1] for y in train[:, -1]])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

    # save model
    model_json = model.to_json()
    with open("lstm_model_rand_epoch.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_rand_epoch.h5")

    return model


def fit_lstm2(train, batch_size, nb_epoch, neurons, timesteps):
    X, y = train[:, 0:-1], np.array([y[-1] for y in train[:, -1]])
    print("\nX shape = ", X.shape)
    print("\ny shape = ", y.shape)
    # print(y)
    model = Sequential()
    # model.add(Input(shape=(X.shape[1], X.shape[2])))  # <--------------------- added this
    model.add(Dense(neurons, input_shape=(X.shape[1], X.shape[2])))  # <--------------------- added this
    # model.add(LSTM(100, activation='tanh', inner_activation='hard_sigmoid', input_shape=(X.shape[1], X.shape[2]),
    #                return_sequences=True)) # <------------------------ instead of '100' was 'neurons'
    model.add(LSTM(100, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True)) # <-----
    # model.add(Dropout(0.3))  # <------------------------------------
    # model.add(LSTM(100, return_sequences=False)) # <----------------
    # model.add(Dropout(0.2))  # <------------------------------------
    model.add(Flatten())  # <--------------------------------------------------
    model.add(Dense(output_dim=1, activation='linear'))

    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=False)

    # save model
    model_json = model.to_json()
    with open("lstm2.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("lstm2.h5")

    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(batch_size, len(X), X.shape[1])
    y_pred = model.predict(X, batch_size=batch_size)

    return y_pred[0, 0]


# run a repeated experiment
def experiment(series, timesteps, shift):
    num_col = len(series.columns)
    len_dataset = len(series)
    # transform data to be stationary
    for i, variable in enumerate(series.columns):
        if variable == 'Stock price':
            real_raw_values = copy.deepcopy(series[variable].values)
        raw_values = series[variable].values
        diff_values = difference(raw_values, shift)
        series[variable] = diff_values
    series = series.drop(series.index[list(range(0, shift))])

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(series, shift, timesteps)
    supervised_values = supervised.values[timesteps:, :].reshape(len_dataset - timesteps - shift, timesteps - shift + 1,
                                                                 num_col)
    train, test = supervised_values[0:-30], supervised_values[-30:]

    print("\n Before scale: \n", train.shape)
    print("\n Before scale: \n", test.shape)

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test, timesteps, shift, len_dataset)
    # print(train_scaled) ###########################################################################
    print("\n After scale: \n", train_scaled.shape)
    print("\n After scale: \n", test_scaled.shape)

    # fit the base model

    lstm_model = fit_lstm2(train_scaled, batch_size=1, nb_epoch=1, neurons=260, timesteps=260)

    # load model
    # '''
    # with open('lstm2.json', 'r') as json_file:
    #   loaded_model_json = json_file.read()
    #   lstm_model = model_from_json(loaded_model_json)
    #    # load weights into new model
    #   lstm_model.load_weights("lstm2.h5")
    # '''

    # forecast test dataset
    predictions = []
    for i in range(len(test_scaled)):
        # predict
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        y_predicted = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        y_predicted = scaler.inverse_transform([[0] * (num_col - 1) + [y_predicted]])[0][-1]
        print(y_predicted)
        # invert differencing
        y_predicted = inverse_difference(real_raw_values, y_predicted, len(test_scaled) + shift - i)
        # store forecast
        predictions.append(y_predicted)
    # report performance
    rmse = sqrt(mean_squared_error(real_raw_values[-30:], predictions))

    return rmse, predictions


def relative_per_dif(true, pred):
    true, pred = np.array(true), np.array(pred)
    return sum(abs((pred - true) / true)) / len(true)


def run():
    companies = ['ABMD', 'AMGN', 'BAX', 'BIO', 'AXGN']  # good companies
    ABMD = pd.read_excel('healthcare_data_new.xls', sheet_name='ABMD')
    # Stock price must be last value in the list
    cols = ['Volume', 'P/B', 'P/E', 'EvEBIT', 'Stock price']
    ABMD = ABMD.loc[:, cols]
    # run experiment
    timesteps = 290
    shift = 30
    rmse, predictions = experiment(ABMD, timesteps, shift)
    ABMD = pd.read_excel('healthcare_data_new.xls', sheet_name='ABMD')

    visualize_stock_prices(ABMD['Stock price'].values[-30:], np.array(predictions))
    visualize_stock_prices(ABMD['Stock price'].values[-60:-30], np.array(predictions))


if __name__ == '__main__':
    run()
