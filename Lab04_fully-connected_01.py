import datetime
import deepchem as dc
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# tf.keras.utils.set_random_seed(456)

logdir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_12outputs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

print(test_w)
# Remove extra (unnecessary) tasks

d = 1024
n_hidden = 500
l_rate = .001
n_epochs = 500
b_size = 100
dropout = 0.50

model = tf.keras.Sequential([
    tf.keras.Input(shape=(d,), dtype=tf.float32, name='x'),
    tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden_layer'),
    tf.keras.layers.Dropout(rate=dropout),
    tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden_layer2'),
    tf.keras.layers.Dropout(rate=dropout),
    tf.keras.layers.Dense(12, activation='sigmoid', name='output')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate), loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(
    train_X, train_y,
    epochs=n_epochs,
    batch_size=b_size,
    callbacks=[tensorboard_callback]
)

test_y_pred = tf.round(model.predict(test_X))

pred_sum = 0
for i in range(0, len(test_y)):
    partial_sum=0
    for j in range(0, 12):
        partial_sum += abs(test_y_pred[i][j] - test_y[i][j])
    if partial_sum != 0:
        pred_sum +=1

print(len(test_y))
print(pred_sum)
