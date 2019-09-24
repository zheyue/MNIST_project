# Use an official Python runtime as a parent image
FROM python:3.5

# Copy the current directory contents into the container at /app
ADD ./MNIST /code

WORKDIR /code

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 4000 available to the world outside this container
EXPOSE 5000

# Define environment variable


# Run app.py when the container launches
CMD ["python", "call.py"]