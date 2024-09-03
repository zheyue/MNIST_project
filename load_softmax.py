from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image

import argparse
import sys
import numpy as np
import os
import gzip
import struct
import cv2

# Import MNIST dataset loader from TensorFlow examples
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# File paths for the MNIST dataset (images and labels)
train_images_file = "MNIST_data/train-images-idx3-ubyte.gz"
train_labels_file = "MNIST_data/train-labels-idx1-ubyte.gz"
t10k_images_file = "MNIST_data/t10k-images-idx3-ubyte.gz"
t10k_labels_file = "MNIST_data/t10k-labels-idx1-ubyte.gz"

def read32(bytestream):
    """Read a 32-bit integer from the given bytestream in big-endian order."""
    dt = np.dtype(np.int32).newbyteorder('>')
    data = bytestream.read(4)  # Read 4 bytes (32 bits)
    return np.frombuffer(data, dt)[0]

def read_labels(filename):
    """Read and one-hot encode the labels from the MNIST dataset."""
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)  # Read magic number
        numberOfLabels = read32(bytestream)  # Read the number of labels
        labels = np.frombuffer(bytestream.read(numberOfLabels), np.uint8)  # Read label data
        # Convert labels to one-hot encoding
        data = np.zeros((numberOfLabels, 10))
        for i in range(len(labels)):
            data[i][labels[i]] = 1
        bytestream.close()
    return data

def read_images(filename):
    """Read and preprocess the images from the MNIST dataset."""
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)  # Read magic number
        numberOfImages = read32(bytestream)  # Read the number of images
        rows = read32(bytestream)  # Read the number of rows (height)
        columns = read32(bytestream)  # Read the number of columns (width)
        # Read image data and reshape into a 2D array (number of images, pixels per image)
        images = np.frombuffer(bytestream.read(numberOfImages * rows * columns), np.uint8)
        images.shape = (numberOfImages, rows * columns)
        # Normalize image pixel values to [0, 1]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        bytestream.close()
    return images

# Load training and testing data from MNIST dataset
train_labels = read_labels(train_labels_file)
train_images = read_images(train_images_file)
test_labels = read_labels(t10k_labels_file)
test_images = read_images(t10k_images_file)

# Alternative way of loading MNIST data using TensorFlow
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# TensorFlow model setup
x = tf.placeholder("float", [None, 784.])  # Placeholder for input data (28x28 image flattened)
W = tf.Variable(tf.zeros([784., 10.]))  # Weights initialized to zero
b = tf.Variable(tf.zeros([10.]))  # Biases initialized to zero
y = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax output for classification
y_ = tf.placeholder("float")  # Placeholder for true labels (one-hot encoded)

# Cross-entropy loss function
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Gradient descent optimizer for training
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initialize all variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# TensorFlow Saver object to save and restore model checkpoints
saver = tf.train.Saver()

# Training loop: perform 600 iterations of training with batch size 100
for i in range(600):
    batch_xs = train_images[100 * i:100 * i + 100]  # Select batch of images
    batch_ys = train_labels[100 * i:100 * i + 100]  # Select batch of labels
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # Train the model on the batch

# Additional training for another 600 iterations to improve accuracy
for i in range(600):
    batch_xs = train_images[100 * i:100 * i + 100]
    batch_ys = train_labels[100 * i:100 * i + 100]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate the model accuracy on the test dataset
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # Check if predictions match true labels
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # Calculate accuracy
print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))  # Print the accuracy

def prediction(input_img):
    """Make a prediction for a single input image."""
    im = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE).astype(np.float32)  # Read the image as grayscale
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)  # Resize the image to 28x28 pixels
    img_gray = (im - (255 / 2.0)) / 255  # Normalize pixel values
    x_img = np.reshape(img_gray, [-1, 784])  # Flatten the image
    output = sess.run(y, feed_dict={x: x_img})  # Make prediction using the trained model
    print('The prediction is: ', np.argmax(output))  # Print the predicted class
    return str(np.argmax(output))

# Test the prediction function with an example image
prediction('MNIST_data/Image/three.png')

# Restore the trained model from a checkpoint and evaluate accuracy
with tf.Session().as_default() as sess:
    saver.restore(sess, 'MNIST_data/model.ckpt')  # Restore model checkpoint
    print('Model loaded successfully.')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # Check if predictions match true labels
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Calculate accuracy
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # Print the accuracy on test data
