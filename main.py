import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers


def import_data():
    # BTC-USD.csv Bitcoin Historical Data (apr2016-nov2021)
    data = pd.read_csv('BTC-USD.csv', parse_dates=["Date"], index_col="Date")

    # We're only interested in the open value
    data = data[:][["Open"]]

    return data


# Use the data from the previous "length" days to predict the price at that time
def preprocess_data(data, length):
    data = data.dropna()

    data = data.iloc[:, 0]
    hist = []
    target = []

    for i in range(len(data) - length):
        x = data[i:i + length]
        y = data[i + length]
        hist.append(x)
        target.append(y)

    # convert list to array
    hist = np.array(hist)
    target = np.array(target).reshape(-1, 1)

    return hist, target


def normalize_data(hist, target, length):
    sc = MinMaxScaler()
    hist_scaled = sc.fit_transform(hist)
    target_scaled = sc.fit_transform(target)

    hist_scaled = hist_scaled.reshape((len(hist_scaled), length, 1))

    return hist_scaled, target_scaled, sc


def split_data(x, y, train_perc):
    n_train = int(x.shape[0] * train_perc)

    x_train = x[:n_train, :, :]
    x_test = x[n_train:, :, :]

    y_train = y[:n_train, :]
    y_test = y[n_train:, :]

    return x_train, x_test, y_train, y_test


def build_model(hidden_units, length, dropout_rate):
    model = tf.keras.Sequential()

    model.add(layers.LSTM(units=hidden_units, return_sequences=True,
                          input_shape=(length, 1), dropout=dropout_rate))

    model.add(layers.LSTM(units=hidden_units, return_sequences=True,
                          dropout=dropout_rate))

    model.add(layers.LSTM(units=hidden_units, dropout=dropout_rate))

    model.add(layers.Dense(units=1))

    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_model(mod, x_train, y_train, epochs, graph=False):
    history = mod.fit(x_train, y_train, epochs=epochs, batch_size=32)

    if graph:
        loss = history.history['loss']
        epoch_count = range(1, len(loss) + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(epoch_count, loss, 'r--')
        plt.legend(['Training Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show();


def evaluate_model(mod, x_test, y_test):
    score = mod.evaluate(x_test, y_test, verbose=0)
    return score


def make_predictions(mod, x_test, y_test, graph):
    pred = mod.predict(x_test)

    if graph:
        plt.figure(figsize=(12, 8))
        plt.plot(y_test, color='blue', label='Real')
        plt.plot(pred, color='red', label='Prediction')
        plt.title('BTC Price Prediction')
        plt.legend()
        plt.show()

    return pred


def denormalize_data(pred, y_test, sc):
    pred_transformed = sc.inverse_transform(pred)
    y_test_transformed = sc.inverse_transform(y_test)

    plt.figure(figsize=(12, 8))
    plt.plot(y_test_transformed, color='blue', label='Real')
    plt.plot(pred_transformed, color='red', label='Prediction')
    plt.title('BTC Price Prediction')
    plt.legend()
    plt.show(0)


def recover_best_parameters(params):
    return params['Model'], params['Scaler'], params['Parameters']


if __name__ == "__main__":
    tr_perc = 0.8
    length_parameters = [80, 90, 100]
    units_parameters = [32, 64, 128]
    dropout_parameters = [0.0, 0.2, 0.3]
    epochs_parameters = [20, 30, 40]
    scores = {}
    best_loss = float('inf')
    best_model = {}
    file =open("scores-log.txt", "w")

    btc_data = import_data()

    for leng in length_parameters:
        btc_history, btc_target = preprocess_data(btc_data, leng)
        btc_hist_scaled, btc_target_scaled, scaler = normalize_data(btc_history, btc_target, leng)
        X_tr, X_te, y_tr, y_te = split_data(btc_hist_scaled, btc_target_scaled, tr_perc)

        for epochs in epochs_parameters:
            for units in units_parameters:
                for dropout in dropout_parameters:
                    btc_model = build_model(units, leng, dropout)
                    train_model(btc_model, X_tr, y_tr, epochs)
                    model_loss = evaluate_model(btc_model, X_te, y_te)
                    parameters = 'Length: ' + str(leng) + \
                                 'Epochs: ' + str(epochs) + \
                                 'Units' + str(units) + \
                                 'Dropout: ' + str(dropout)
                    scores[parameters] = model_loss
                    file.write(str(model_loss))
                    file.write(parameters)
                    if model_loss < best_loss:
                        best_loss = model_loss
                        best_model = {
                            'Model': btc_model,
                            'Scaler': scaler,
                            'Parameters': parameters
                        }

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    print(dict(sorted_scores))

    print('***** BEST MODEL *****')
    btc_model, scaler, parameters = recover_best_parameters(best_model)
    print(parameters)

    btc_pred = make_predictions(btc_model, X_te, y_te, False)
    denormalize_data(btc_pred, y_te, scaler)

    file.close()