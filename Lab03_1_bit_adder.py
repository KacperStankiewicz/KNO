import tensorflow as tf
import numpy as np
import datetime

logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-adam-1000-binary-8'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),  # Input layer with 3 features (a , b and c)
    tf.keras.layers.Dense(10, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(8, activation='sigmoid')  # Output layer with 2 labels (result + carry)
])

# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_train = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
], dtype=np.float32)
y_train = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=np.float32)
model.fit(x_train, y_train, epochs=3000, callbacks=[tensorboard_callback])

loss, accuracy = model.evaluate(x_train, y_train)

predictions = model.predict(x_train)
print(predictions)
mapped_array = [[1 if elem > 0.5 else 0 for elem in inner_list] for inner_list in predictions]

for e in mapped_array2:
    if e == [1, 0, 0, 0, 0, 0, 0, 0]:
        print([0,0])