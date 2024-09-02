
# MNIST Handwritten Digit Recognition Project

This project demonstrates the use of TensorFlow for training a model to recognize handwritten digits from the MNIST dataset. Additionally, it showcases experience in deploying a simple prediction service using Flask, Docker, and Cassandra for storing user-uploaded data. 

The project was primarily built as a learning exercise to gain hands-on experience with these tools.

## Project Background

The goal of this project was to build a basic machine learning model to recognize handwritten digits using the MNIST dataset. It also involved creating a RESTful API using Flask to allow users to submit images for digit recognition and store the results in a Cassandra database.

While this project was not intended to achieve state-of-the-art accuracy, it served as an opportunity to work with TensorFlow, Docker, and Cassandra, and to build a simple end-to-end machine learning application.

## Problem Definition

The task is to classify grayscale images of handwritten digits (0-9) from the MNIST dataset. The challenge involves:

1. Training a machine learning model using TensorFlow on the MNIST dataset.
2. Deploying the model via a Flask API to allow users to upload images for prediction.
3. Storing user-submitted data and predictions in a Cassandra database for future analysis.

## Solution Overview

### Model Training
The model was built using TensorFlow and trained on the MNIST dataset. After training, the model is saved as a checkpoint to be loaded later for making predictions.

### Flask API
A Flask-based API was developed to handle user input. Users can upload images, which are processed and passed through the trained model to return predictions.

### Cassandra Integration
The results of user-submitted images and predictions are stored in a Cassandra database to maintain a record of interactions.

## Project Structure

```
MNIST_Project/
│
├── MNIST_data/                 # Directory containing the MNIST dataset
├── mnist_softmax.py            # Script for training the model
├── load_softmax.py             # Script for loading the trained model
├── call.py                     # Flask API implementation
├── connect.py                  # Cassandra database connection and query scripts
└── Dockerfile                  # Docker configuration for setting up the environment
```

## How to Run the Project

### Prerequisites

Before running the project, make sure you have the following dependencies installed:

- Docker
- TensorFlow 1.14.0
- Flask 1.0.2
- Cassandra
- Python packages listed in `requirements.txt`

### Steps to Run

1. **Set up the TensorFlow environment:**

   ```bash
   $ conda activate tf
   ```

2. **Train the model:**

   Navigate to the project directory and run:

   ```bash
   $ python mnist_softmax.py
   ```

   This script will train the model on the MNIST dataset and save the model checkpoint to `MNIST_data/model.ckpt`.

3. **Run the Flask API:**

   Start the Flask server to serve the model for predictions:

   ```bash
   $ python call.py
   ```

4. **Set up the Cassandra Database:**

   Launch a Cassandra container using Docker:

   ```bash
   $ docker run --name mnist-cassandra -p 9042:9042 -d cassandra:latest
   ```

5. **Connect Flask with Cassandra:**

   Make sure the Cassandra database is connected by running:

   ```bash
   $ docker exec -it mnist-cassandra cqlsh
   ```

   Create the necessary keyspace and table using `connect.py`.

6. **Upload an Image for Prediction:**

   Use the following curl command to upload an image for prediction:

   ```bash
   $ curl -XPOST http://127.0.0.1:5000/upload -F "file=@path_to_image.png"
   ```

   Alternatively, you can use the web interface provided by the Flask application.

## Technical Details

### TensorFlow Model
- **Model:** A simple softmax classifier trained on the MNIST dataset.
- **Training:** The model is trained using stochastic gradient descent (SGD) and the softmax cross-entropy loss function.
- **Saving:** The trained model is saved using TensorFlow's `Saver()` class to be reused during prediction.

### Flask API
- **Endpoints:** The main endpoint `/upload` accepts image uploads and returns the model's prediction in JSON format.
- **Image Processing:** Uploaded images are resized and normalized before being passed to the model for prediction.

### Cassandra Database
- **Setup:** Cassandra is used to store user-uploaded images and their corresponding predictions.
- **Schema:** The database schema consists of a simple table with fields for the image file path, prediction, and timestamp.

## Future Improvements

- **Model Enhancement:** Experiment with more complex neural networks, such as convolutional neural networks (CNNs), to improve prediction accuracy.
- **Scalability:** Deploy the Flask API and Cassandra database in a cloud environment for better scalability and availability.
- **Web Interface:** Build a more user-friendly web interface for uploading images and displaying results.

## Dependencies

- TensorFlow 1.14.0
- Flask 1.0.2
- Docker
- Cassandra
- Other Python packages (see `requirements.txt`)
