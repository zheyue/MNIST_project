from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Load MNIST dataset
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def main(_):
    # Define the model
    x = tf.placeholder(tf.float32, [None, 784])  # Input placeholder for images
    W = tf.Variable(tf.zeros([784, 10]))  # Weight variable
    b = tf.Variable(tf.zeros([10]))  # Bias variable
    y = tf.matmul(x, W) + b  # Linear model for output

    # Placeholder for true labels
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Cross-entropy loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Gradient Descent optimizer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Start a TensorFlow session
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()  # For saving the model
    tf.global_variables_initializer().run()  # Initialize variables

    # Training loop
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # Get next batch of 100 images and labels
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # Run training step

    # Save the model after training
    saver.save(sess, 'MNIST_data/model.ckpt')

    # Evaluate the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # Compare predicted labels and true labels
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Calculate accuracy
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # Print accuracy on test set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
