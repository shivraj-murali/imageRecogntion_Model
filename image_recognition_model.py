import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
import cv2


# load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


########## DATA CLEANING AND LABELLING ###################
def image_cleaning(image, threshold_value=150):

    # Apply binary thresholding
    _, thresholded_image = cv2.threshold(
        image, 150, 255, cv2.THRESH_BINARY)

    return thresholded_image


X_train_processed = np.array(
    [image_cleaning(image, 150) for image in X_train])

# Preprocess the test images
X_test_processed = np.array(
    [image_cleaning(image, 150) for image in X_test])


# we divide by 255 to normalise the data
X_valid, X_train = X_train_processed[:5000]/255.0, X_train_processed[5000:]/255.0
y_valid, y_train = y_train[:5000], y_train[5000:]

class_names = ['Zero', 'One', 'Two', 'Three',
               'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']


########## MODEL BUILDING ##############
# Steps
# 1 - Flatten the datapoint
# 2 - Create Hidden Layers
# 3 - Create output layer and make the prediction

model = keras.models.Sequential([
    # Flatten images into a input layer of 784 pixels
    keras.layers.Flatten(input_shape=[28, 28]),
    # We use dense as we are using a fully connected neural network activation function is relu we have 300 neurons
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    # This represents the output layer
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

image_recognition_model = model.fit(
    X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))