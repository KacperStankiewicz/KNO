import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_diabetes
import pandas as pandasForSortingCSV
import csv

logdir = "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch')

input_file = 'data/pima-indians-diabetes.csv'
data = np.loadtxt(input_file, delimiter=',')
dataX, dataY = data[:, :-1], data[:, -1]

train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, random_state=5)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio),
                                                random_state=5)


def train_model(model, num_of_epochs, batch_size):
    model.fit(
        x_train, y_train,
        epochs=num_of_epochs,
        batch_size=batch_size,
        callbacks=tensorboard_callback,
        verbose=False
    )

    loss, accuracy = model.evaluate(x_val, y_val)
    return accuracy


def build_network(num_of_layers, num_of_neurons, activation, dropout_rate, learning_rate, num_of_epochs, batch_size):
    model = tf.keras.Sequential([tf.keras.Input(shape=(8,), dtype=tf.float32, name='input')])
    for i in range(num_of_layers + 1):
        model.add(tf.keras.layers.Dense(num_of_neurons, activation=activation))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation=activation, name='output'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return train_model(model, num_of_epochs, batch_size)


def grid_select():
    header = ['num_of_layers', 'num_of_neurons', 'activation', 'dropout_rate', 'learning_rate', 'num_of_epochs',
              'batch_size', 'accuracy']

    f = open('data/results.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(header)

    num_of_layers = range(1, 6, 1)
    num_of_neurons = [50, 100, 150]
    activation = ['relu', 'sigmoid']
    dropout_rate = np.arange(0.1, 1, 0.2)
    learning_rate = [0.001]
    num_of_epochs = [50]
    batch_size = [100]

    best_params = []
    best_acc = 0

    for layers in num_of_layers:
        for neurons in num_of_neurons:
            for activ in activation:
                for dropout in dropout_rate:
                    for learning in learning_rate:
                        for epochs in num_of_epochs:
                            for batch in batch_size:
                                acc = build_network(layers, neurons, activ, dropout, learning, epochs, batch)
                                writer.writerow([layers, neurons, activ, dropout, learning, epochs, batch, acc])

                                if acc > best_acc:
                                    best_acc = acc
                                    best_params = [layers, neurons, activ, dropout, learning, epochs, batch]

    f.close()
    print(best_params, best_acc)


def random_select(iter=10):
    params = {
        'num_of_layers': range(1, 6, 1),
        'num_of_neurons': [50, 100, 150],
        'activation': ['relu', 'sigmoid'],
        'dropout_rate': np.arange(0.1, 1, 0.2),
        'learning_rate': [0.001],
        'num_of_epochs': [50],
        'batch_size': [100]
    }
    best_params = None
    best_acc = 0

    param_samples = list(ParameterSampler(
        params, n_iter=iter, random_state=5
    ))

    for parameters in param_samples:
        acc = build_network(
            parameters['num_of_layers'],
            parameters['num_of_neurons'],
            parameters['activation'],
            parameters['dropout_rate'],
            parameters['learning_rate'],
            parameters['num_of_epochs'],
            parameters['batch_size']
        )

        if acc > best_acc:
            best_acc = acc
            best_params = parameters

    print(best_params)
    print(best_acc)


if __name__ == "__main__":
    # grid_select()
    random_select()
