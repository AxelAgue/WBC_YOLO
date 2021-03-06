import tensorflow as tf
import xml.etree.ElementTree as ET
import cv2
import numpy as np

import utils


"""
from Load_Labels import give_label
from Load_Labels import labels_total

num_image1 = "007"
num_image2 = "006"

image = cv2.imread("C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00" + num_image1 + ".jpg")

tree1 = ET.parse("C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + num_image1 + ".xml")
tree2 = ET.parse("C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + num_image2 + ".xml")





## imagen 1
label_test1 = give_label(tree1)
total1 = labels_total(label_test1)

## imagen 2
label_test2 = give_label(tree2)
total2 = labels_total(label_test2)

testtt = loss_function(total1, total2, n_grid_cells=10)

sess = tf.Session()


print(sess.run(testtt))
"""

def loss_function(alpha, beta, n_grid_cells):

    original_label = tf.reshape(alpha, [-1, 100, 8])
    yolo_label = tf.reshape(beta, [-1, 100, 8])

    ## tf.reshape(y_conv, [-1, 10, 10, 8])

    ## uso mismos valores que el paper

    l_noobj = 5.0
    l_coord = 0.5

    # Get the predicted values
    x_hat = utils.list_slice(yolo_label, 5 * _j + 0, axis=3)
    y_hat = utils.list_slice(yolo_label, 5 * _j + 1, axis=3)
    w_hat = utils.list_slice(yolo_label, 5 * _j + 2, axis=3)
    h_hat = utils.list_slice(yolo_label, 5 * _j + 3, axis=3)
    C_hat = utils.list_slice(yolo_labeld, 5 * _j + 4, axis=3)
    p_hat = pred[..., 5 * _B:]

    p_hat = []
    x_hat = []
    y_hat = []
    w_hat = []
    h_hat = []
    c1_hat = []
    c2_hat = []
    c3_hat = []

    for k in range(n_grid_cells*n_grid_cells):

        aux = yolo_label[k]

        p_hat.append(aux[0])
        x_hat.append(aux[1])
        y_hat.append(aux[2])
        w_hat.append(aux[3])
        h_hat.append(aux[4])
        c1_hat.append(aux[5])
        c2_hat.append(aux[6])
        c3_hat.append(aux[7])


    p = []
    x = []
    y = []
    w = []
    h = []
    c1 = []
    c2 = []
    c3 = []

    for k in range(n_grid_cells*n_grid_cells):

        aux2 = original_label[k]

        p.append(aux2[0])
        x.append(aux2[1])
        y.append(aux2[2])
        w.append(aux2[3])
        h.append(aux2[4])
        c1.append(aux2[5])
        c2.append(aux2[6])
        c3.append(aux2[7])


    x_err = tf.square(np.asarray(x) - np.asarray(x_hat))
    y_err = tf.square(np.asarray(y) - np.asarray(y_hat))
    w_err = tf.square(tf.sqrt(np.asarray(w)) - tf.sqrt(np.asarray(w_hat)))
    h_err = tf.square(tf.sqrt(np.asarray(h)) - tf.sqrt(np.asarray(h_hat)))
    c1_err = tf.square(np.asarray(c1) - np.asarray(c1_hat))
    c2_err = tf.square(np.asarray(c2) - np.asarray(c2_hat))
    c3_err = tf.square(np.asarray(c3) - np.asarray(c3_hat))
    p_err = tf.square(np.asarray(p) - np.asarray(p_hat))

    loss = tf.reduce_sum(

        tf.reduce_sum(
             (
                l_coord * (x_err + y_err + w_err + h_err)
                + c1_err + c2_err + c3_err + p_err
            )

    ))

    return np.asarray(loss)

"""
from Load_Labels import give_label
from Load_Labels import labels_total

num_image1 = "007"
num_image2 = "006"

image = cv2.imread("C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00" + num_image1 + ".jpg")

tree1 = ET.parse("C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + num_image1 + ".xml")
tree2 = ET.parse("C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + num_image2 + ".xml")





## imagen 1
label_test1 = give_label(tree1)
total1 = labels_total(label_test1)

## imagen 2
label_test2 = give_label(tree2)
total2 = labels_total(label_test2)

testtt = loss_function(total1, total2, n_grid_cells=10)

print(total1.shape)

"""

#######

"""
####
W_conv2 = weight_variable([7, 7, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([7, 7, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([7, 7, 128, 256])
b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_fc1 = weight_variable([15 * 15 * 256, 1024])
b_fc1 = weight_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 15 * 256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
####
"""

