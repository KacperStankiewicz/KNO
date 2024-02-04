import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN
from tensorflow.keras.models import Sequential


def create_sequences(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def get_lstm_model(n_steps, X_train, y_train):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model


def get_gru_model(n_steps, X_train, y_train):
    model = Sequential()
    model.add(GRU(100, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model


def get_simple_model(n_steps, X_train, Y_train):
    model = Sequential()
    model.add(SimpleRNN(100, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model


def perform_rnn(col, df, type):
    data = df[col].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    n_steps = 7

    # Create sequences
    X, y = create_sequences(data_normalized, n_steps)

    # Split the data into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test, Y_train, Y_test = X[:split], X[split:], y[:split], y[split:]

    # Reshape input data to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = None
    if type == 'LSTM':
        model = get_lstm_model(n_steps, X_train, Y_train)
    elif type == 'GRU':
        model = get_gru_model(n_steps, X_train, Y_train)
    elif type == 'SIMPLE':
        model = get_simple_model(n_steps, X_train, Y_train)
    else:
        print('ERROR')

    # Train model
    model.fit(X_train, Y_train, epochs=50, verbose=0)
    # Evaluate the model on the test set
    mse = model.evaluate(X_test, Y_test, verbose=0)
    print(f'Mean Squared Error on Test Set: {mse}')

    # Make predictions
    predictions = model.predict(X_test)

    # Plot actual vs. predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test, label=f'Actual {col} Prices', color='black')
    plt.plot(predictions, label=f'Predicted {col} Prices', color='green')
    plt.title(f'{type}, Actual vs. Predicted {col} Prices')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{col} Price')
    plt.legend()
    plt.show()


df = pd.read_csv('data.csv', parse_dates=True)

perform_rnn('High', df, 'LSTM')
perform_rnn('High', df, 'GRU')
perform_rnn('High', df, 'SIMPLE')
perform_rnn('Low', df, 'LSTM')
perform_rnn('Low', df, 'GRU')
perform_rnn('Low', df, 'SIMPLE')
perform_rnn('Close', df, 'LSTM')
perform_rnn('Close', df, 'GRU')
perform_rnn('Close', df, 'SIMPLE')
perform_rnn('Volume', df, 'LSTM')
perform_rnn('Volume', df, 'GRU')
perform_rnn('Volume', df, 'SIMPLE')
