# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A deep MNIST classifier using convolutional layers.

This script builds and trains a convolutional neural network (CNN) to classify 
handwritten digits from the MNIST dataset. It also provides functionality to 
test the model's accuracy on the test dataset.

For extensive documentation, refer to:
https://www.tensorflow.org/get_started/mnist/pros
"""

# Disable linter warnings to maintain consistency with the tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

# Import the MNIST dataset handling utilities from TensorFlow examples.
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def deepnn(x):
    """
    deepnn builds the graph for a deep convolutional neural network for 
    classifying digits.

    Args:
        x: A placeholder tensor of shape (N_examples, 784), where 784 represents
           the number of pixels in a flattened 28x28 MNIST image.
           
    Returns:
        A tuple of:
            - y_conv: A tensor of shape (N_examples, 10) containing the logits 
              for digit classification into 10 classes (digits 0-9).
            - keep_prob: A scalar placeholder for the dropout probability during
              training.
    """
    # Reshape the input to 4D tensor: [batch_size, height, width, channels].
    # MNIST images are 28x28 pixels and grayscale (1 channel).
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer: Applies 32 filters of size 5x5 to the input image.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # First pooling layer: Downsamples the image by a factor of 2 (max pooling).
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer: Applies 64 filters of size 5x5 to the pooled output.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer: Further downsamples the image by a factor of 2.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer: After pooling, the image size is reduced to 7x7x64. 
    # This layer flattens the 7x7x64 output and maps it to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout: Randomly drops units from the network during training to prevent overfitting.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer: Maps the 1024 features to 10 classes (one for each digit).
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """
    conv2d returns a 2D convolution layer with full stride.
    
    Args:
        x: Input tensor.
        W: Filter tensor (weights).
    
    Returns:
        The output tensor after applying the convolution operation.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    max_pool_2x2 performs max pooling with a 2x2 filter and stride of 2.
    
    Args:
        x: Input tensor.
    
    Returns:
        The downsampled output tensor after max pooling.
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """
    weight_variable initializes a weight variable of a given shape with a truncated normal distribution.
    
    Args:
        shape: The shape of the weight tensor.
    
    Returns:
        A weight tensor initialized with a truncated normal distribution.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    bias_variable initializes a bias variable with a constant value.
    
    Args:
        shape: The shape of the bias tensor.
    
    Returns:
        A bias tensor initialized to a constant value (0.1).
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Load the MNIST dataset, which contains handwritten digit images.
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Placeholder for input images, each image is represented as a flattened 784-pixel vector.
    x = tf.placeholder(tf.float32, [None, 784])

    # Placeholder for the true labels, represented as one-hot encoded vectors.
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the CNN graph for digit classification.
    y_conv, keep_prob = deepnn(x)

    # Define the loss function: softmax cross-entropy between predicted logits and true labels.
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    # Define the optimizer: Adam optimizer with a learning rate of 1e-4.
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Calculate accuracy by comparing predicted labels to the true labels.
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Save the computation graph to a temporary directory for TensorBoard visualization.
    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    # Start a TensorFlow session to run the training process.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            # Fetch the next training batch (50 samples at a time).
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                # Print the training accuracy every 100 steps.
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            # Run the training step with dropout applied.
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        # After training, evaluate the model on the test dataset.
        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    # Parse command-line arguments for data directory.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    # Run the main function, passing any additional arguments from the command line.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
