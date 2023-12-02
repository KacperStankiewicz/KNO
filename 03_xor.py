import tensorflow as tf
import numpy as np
import datetime

logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),  # Input layer with 2 features (x1 and x2)
    tf.keras.layers.Dense(10, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 label
])

#model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_train = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
y_train = np.array([0, 1, 0, 1], dtype=np.float32)
model.fit(x_train, y_train, epochs=300,callbacks=[tensorboard_callback])

loss, accuracy = model.evaluate(x_train, y_train)

predictions = model.predict(x_train)
print(predictions)
print(list(map(lambda e : 1 if e > 0.5 else 0, predictions)))
print(list(zip(x_train,list(map(lambda e : 1 if e > 0.5 else 0, predictions)))))
