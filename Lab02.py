import tensorflow as tf
import math, sys, getopt
import numpy as np

def rotate(x, y, degrees):
    radians = math.radians(degrees)
    rotation_matrix = tf.constant([math.cos(radians),
                                   -math.sin(radians),
                                   math.sin(radians),
                                   math.cos(radians)],
                                  shape=[2, 2])

    coordinates = tf.constant([x, y], shape=[2, 1])
    coordinates = tf.cast(coordinates, dtype=tf.float32)

    return tf.matmul(rotation_matrix, coordinates).numpy()

def validate(n,a,b):
    if n * n != len(a) or n != len(b):
        print("invalid shape!")
        exit(1)

    a = tf.constant(a, dtype=tf.float32, shape=[n, n])
    b = tf.constant(b, dtype=tf.float32, shape=[n, 1])

    if tf.linalg.det(a) == 0:
        print("not possible to solve")
        exit(1)
    return a,b

@tf.function
def calculate_soe(a, b):
    inversed_a = tf.linalg.inv(a)
    return tf.matmul(inversed_a, b)


# zadanie 1
# radians = math.radians(90)
# rotation_matrix = tf.constant([math.cos(radians),
#                                -math.sin(radians),
#                                math.sin(radians),
#                                math.cos(radians)],
#                               shape=[2, 2])
#
# coordinates = tf.constant([5., 5.], shape=[2, 1])
#
# print(tf.matmul(rotation_matrix, coordinates).numpy())

# zadanie 2
# wynik = rotate(2, 2, 30)
# print(wynik)

# zadanie 3
# a = tf.constant([2, -1, 1, 1, -1, 2, 5, -2, 2], dtype=tf.float32, shape=[3, 3])
# b = tf.constant([7, 6, 15], dtype=tf.float32, shape=[3, 1])
# inversed_a = tf.linalg.inv(a)
#
# print(tf.matmul(inversed_a, b))

# zadanie 4
n = int(sys.argv[1])
a = np.array(sys.argv[2].split(',')).astype(int)
b = np.array(sys.argv[3].split(',')).astype(int)

a,b = validate(n,a,b)

print(calculate_soe(a, b))
