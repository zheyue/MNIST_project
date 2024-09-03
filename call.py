import tensorflow as tf
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import connect  # Custom module for database connection and operations
import time
import logging

# Define a placeholder for input MNIST images, flattened to 784 pixels
x = tf.placeholder(tf.float32, [None, 784])

# Define weight matrix W and bias vector b for 10 classes (digits 0-9)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Use softmax activation to compute the probability distribution over classes
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Create a TensorFlow session and restore the pre-trained model
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'MNIST_data/model.ckpt')  # Load the saved model parameters

# Create a Flask web application
app = Flask(__name__)

@app.route('/upload', methods=['POST', 'GET'])
def prepare_image():
    """Process the uploaded image file, perform prediction, and store the result in the database."""
    # Get the uploaded file
    file = request.files['file']
    filename = secure_filename(file.filename)  # Ensure a secure filename

    # Read the image using OpenCV and convert it to grayscale
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # Resize the image to 28x28 to match the MNIST format
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)
    
    # Normalize the pixel values to the range [-1, 1], preparing it for model input
    img_gray = (im - (255 / 2.0)) / 255
    x_img = np.reshape(img_gray, [-1, 784])

    # Use the TensorFlow model to make a prediction
    output = sess.run(y, feed_dict={x: x_img})
    result = str(np.argmax(output))  # Get the predicted class (0-9)

    # Get the current timestamp and store the result in the database
    t = int(round(time.time() * 1000))
    date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t / 1000))
    connect.insertData(date, filename, result)  # Save prediction result in the database

    # Return the prediction result
    return "The prediction is: " + result


@app.route('/')
def index():
    """Homepage providing a form for file upload."""
    return '''
    <!doctype html>
    <html>
    <body>
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
        <input type='submit' value='Upload'>
    </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # Create Cassandra Keyspace (if not already present)
    connect.createKeySpace()
    
    # Run the Flask application with debugging enabled
    app.run(debug=True)
