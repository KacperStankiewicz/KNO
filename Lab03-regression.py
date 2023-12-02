import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.00001
training_epochs = 1000

# the training dataset
x_train = np.linspace(0, 10, 100)
y_train = (x_train ** 2
           + x_train * np.random.normal(0, 1, 100)
           + np.random.normal(0, 1, 100))

# plot of data
plt.scatter(x_train, y_train)
plt.savefig('datapoints.png')

# declare weights
alpha = tf.Variable(0.)
beta = tf.Variable(1.)
gamma = tf.Variable(2.)


# Define linear regression expression y
def linreg(x):
    y = alpha * x ** 2 + x * beta + gamma
    return y


# Define loss function (MSE)
def squared_error(y_pred, y_true):
    return abs(tf.reduce_mean(tf.square(y_pred - y_true)))


# train model
for epoch in range(training_epochs):
    # Compute loss within Gradient Tape context
    with tf.GradientTape() as tape:
        y_predicted = linreg(x_train)

        loss = squared_error(y_predicted, y_train)

        # Get gradients
        gradients = tape.gradient(loss, [alpha, beta, gamma])

        # Adjust weights
        alpha.assign_sub(gradients[0] * learning_rate)
        beta.assign_sub(gradients[1] * learning_rate)
        gamma.assign_sub(gradients[2] * learning_rate)

    # Print output
    print(f"Epoch count {epoch}: Loss value: {loss.numpy()}")

print(alpha.numpy())
print(beta.numpy())
print(gamma.numpy())

# Plot the best fit line
plt.scatter(x_train, y_train)
plt.plot(x_train, linreg(x_train), 'r')
plt.show()
