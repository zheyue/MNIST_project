import tensorflow as tf
from flask import Flask


import load_softmax
 
app = Flask(__name__)



@app.route('/')
def prepare_image():
    output = load_softmax.prediction('MNIST_data/Image/nine.png')
    return "The prediction is : "+ output

if __name__ == '__main__':
    app.run(debug=True)


