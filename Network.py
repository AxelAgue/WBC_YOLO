## construction of layers

import tensorflow as tf

import xml.etree.ElementTree as ET
import numpy as np

from Load_Labels import abrir_labels
from Load_Images import open_images

from Cost_Function import loss_function


aa = abrir_labels(n_imag=1)
bb = open_images(n_imag=1)

labels = aa.reshape(10, 10, 10, 8)
images = bb.reshape(10, 120, 120, 3)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5, name="Weights")
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name="bias")
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


## initialize placeholders:

x = tf.placeholder(tf.float32, [None, 120, 120, 3], name="input")
y_ = tf.placeholder(tf.float32, [None, 10, 10, 8], name="input")

x_image = tf.reshape(x, shape=[-1, 120, 120, 3])

W_conv1 = weight_variable([7, 7, 1*3, 2*3])
b_conv1 = bias_variable([2*3])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([60 * 60 * 2 * 3, 1024])
b_fc1 = weight_variable([1024])

h_pool2_flat = tf.reshape(h_pool1, [-1, 60 * 60 * 2 * 3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 800])
b_fc2 = bias_variable([800])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

last_layer = tf.reshape(y_conv, [-1, 10, 10, 8])

cost_function = loss_function(y_, last_layer, n_grid_cells=10)
## la funcion espera un valor y le estoy dando un tensor

## cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=last_layer))
## Tensor("Mean:0", shape=(), dtype=float32)

print(cost_function)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(last_layer, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        ## seleccionar batch

        train_step.run(feed_dict={x: images, y_: labels, keep_prob: 0.5})

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: images, y_: labels, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))


