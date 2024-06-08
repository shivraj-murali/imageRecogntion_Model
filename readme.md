# Digit Recognizer Machine Learning Model

## Overview

This project implements a machine learning model to recognize handwritten digits (0-9) from images. The model is trained on the MNIST dataset, a well-known dataset containing 70,000 labeled images of handwritten digits. This README provides a comprehensive guide on the setup, training, evaluation, and usage of the digit recognizer.
Note: Incase you wish to run it on google collab insted of locally running it in your pc upload "my_model.h5" and "image_recognition_model.ipynb" to collab and run the "image_recognition_model.ipynb" file on your desired image

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Using the Model](#using-the-model)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## Features

- **Image Preprocessing**: Normalizes and reshapes input images.
- **Convolutional Neural Network (CNN)**: Utilizes a deep learning model for high accuracy.
- **Training and Evaluation Scripts**: Scripts to train the model and evaluate its performance.
- **Model Saving and Loading**: Functionality to save and load trained models.
- **Prediction Script**: Predicts the digit in a new image.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook (optional, for exploring the dataset and model interactively)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/digit-recognizer.git
    cd digit-recognizer
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Using the Model

1. Ensure you have a webcam to click an image. If you dont have one simply upload a image and name it "opencv_frame_0.png" and run the following scripts
2. Run the training script:
    ```sh
    python imagecapture.py
    python preprocessor.py
    python app.py
    ```
   This script will preprocess the data, build the model, and train it. The trained model will be saved as `my_model.h5`.

### Using the Model

1. To make predictions on new images, use the prediction script:
    ```sh
    python app.py
    ```
   This will output the predicted digit.

## Dataset

The MNIST dataset is used for training and evaluating the model. It consists of 60,000 training images and 10,000 test images of handwritten digits, each 28x28 pixels in grayscale.

## Model Architecture

The model is a Multilayer Perceptron (MLP) with the following layers:
- Input layer: 28x28 color or grayscale images
- First Hidden Layer: 300 neurons ,activation 'relu'
- Second Hidden layer: 100 neurons ,activation 'relu'
- Output layer: 10 units (one for each digit), activation 'softmax'

## Results

The model achieves an accuracy of approximately 98% on the MNIST test dataset.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements or bug fixes. For major changes, please discuss them in an issue first to ensure compatibility.

---

Feel free to contact for any questions or feedback. Happy coding!
shivrajmurali504@gmail.com
