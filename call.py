import tensorflow as tf
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import cv2
import numpy as np
import connect
import time
import logging




log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784., 10.]))
b = tf.Variable(tf.zeros([10.]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,'MNIST_data/model.ckpt')

 
app = Flask(__name__)

@app.route('/upload', methods=['POST','GET'])
def prepare_image():
        file = request.files['file']
        filename=secure_filename(file.filename)
        im = cv2.imread(filename,cv2.IMREAD_GRAYSCALE).astype(np.float32)
        im = cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)
        img_gray = (im - (255 / 2.0)) / 255
        x_img = np.reshape(img_gray , [-1 , 784])
        output = sess.run(y , feed_dict = {x:x_img})
        result = str(np.argmax(output))
        t = int(round(time.time()*1000))
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t/1000))
        connect.insertData(date, filename,result)
        return "The prediction is : " +result



@app.route('/')
def index():
	return '''
	<!doctype html>
	<html>
	<body>
	<form action='/upload' method='post' enctype='multipart/form-data'>
  		<input type='file' name='file'>
	<input type='submit' value='Upload'>
	</form>
	'''    
if __name__ == '__main__':
    connect.createKeySpace();    
    app.run(debug=True)

