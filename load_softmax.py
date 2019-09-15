from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import gc
from keras import backend as K



import argparse
import sys
import numpy as np
import os
import gzip
import struct
import cv2

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

train_images_file = "MNIST_data/train-images-idx3-ubyte.gz"
train_labels_file = "MNIST_data/train-labels-idx1-ubyte.gz"
t10k_images_file = "MNIST_data/t10k-images-idx3-ubyte.gz"
t10k_labels_file = "MNIST_data/t10k-labels-idx1-ubyte.gz"

def read32(bytestream):
    # 由于网络数据的编码是大端，所以需要加上>
    dt = np.dtype(np.int32).newbyteorder('>')
    data = bytestream.read(4)
    return np.frombuffer(data, dt)[0]

def read_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)
        numberOfLabels = read32(bytestream)
        labels = np.frombuffer(bytestream.read(numberOfLabels), np.uint8)
        data = np.zeros((numberOfLabels, 10))
        for i in range(len(labels)):
            data[i][labels[i]] = 1
        bytestream.close()
    return data

def read_images(filename):
    # 把文件解压成字节流
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)
        numberOfImages = read32(bytestream)
        rows = read32(bytestream)
        columns = read32(bytestream)
        images = np.frombuffer(bytestream.read(numberOfImages * rows * columns), np.uint8)
        images.shape = (numberOfImages, rows * columns)
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        bytestream.close()
    return images

train_labels = read_labels(train_labels_file)
train_images = read_images(train_images_file)
test_labels = read_labels(t10k_labels_file)
test_images = read_images(t10k_images_file)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


x = tf.placeholder("float", [None, 784.])
W = tf.Variable(tf.zeros([784., 10.]))
b = tf.Variable(tf.zeros([10.]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float")
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

for i in range(600):
    batch_xs = train_images[100 * i:100 * i + 100]
    batch_ys = train_labels[100 * i:100 * i + 100]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# 提高准确度，训练多一次
for i in range(600):
    batch_xs = train_images[100 * i:100 * i + 100]
    batch_ys = train_labels[100 * i:100 * i + 100]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))


def prediction(input_img):
    im = cv2.imread(input_img,cv2.IMREAD_GRAYSCALE).astype(np.float32)
    im = cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)
    img_gray = (im - (255 / 2.0)) / 255
    x_img = np.reshape(img_gray , [-1 , 784])
    print (x_img)
    output = sess.run(y , feed_dict = {x:x_img})
    print ('the y :   ', '\n',output)
    print ('the predict is : ', np.argmax(output))
    return str(np.argmax(output))


with tf.Session().as_default() as sess: 
  saver.restore(sess, 'MNIST_data/model.ckpt')
  print('LOAD DONE 100%')
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  
  
