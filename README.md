# RSS Value Prediction Using Log Distance and Neural Network

This repository contains a Python implementation for predicting the Received Signal Strength (RSS) values based on log distance using both Linear Regression and a Neural Network model.

### Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Data Description](#data-description)
4. [Methodology](#methodology)
5. [Usage](#usage)
6. [Results](#results)

## Project Overview

The goal of this project is to predict the RSS values using the concept of log distance. The model uses two different approaches:
1. **Linear Regression**: A basic linear model to predict the relationship between log distance and RSS values.
2. **Neural Network**: A deeper neural network model to predict the same relationship with more complexity and flexibility.

The dataset is based on the locations of transmitters and receivers, with corresponding RSS values.

## Prerequisites

To run this project, you need the following libraries:

- `numpy` >= 1.21.0
- `pandas` >= 1.3.0
- `matplotlib` >= 3.4.0
- `tensorflow` >= 2.0.0
- `scikit-learn` >= 0.24.0

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```
## Data Description

The project uses two main input files:

1. **locations.txt**: This file contains the (X, Y) coordinates of the receivers and transmitters. Each row in this file represents the coordinates of a location. These coordinates are used to calculate the log distance between transmitter-receiver pairs.

2. **rss_values.txt**: This file contains the corresponding RSS values for the transmitter-receiver pairs. The values are organized in a matrix format, where rows represent different receivers, and columns represent different transmitters. The matrix entries indicate the signal strength (RSS) between a specific transmitter and receiver. If a value is `np.inf`, it indicates that no valid signal is received between that pair.

Both files are required to be present in the same directory as the Python script for proper execution of the model.

## Methodology

### Log Distance Model
The distance between a receiver and a transmitter is calculated using the Euclidean distance formula:

`Distance = √((x₂ - x₁)² + (y₂ - y₁)²)`

This distance is then converted to a logarithmic scale using the formula:

`Log Distance = -10 * log₁₀(Distance)`


This transformation allows the model to handle the wide range of distances more effectively and aligns with common practices in signal propagation modeling.

### Linear Regression
A simple linear regression model is used to predict the RSS values based on the calculated log distance. The model is implemented using the closed-form solution for linear regression:

`β = (Xᵀ X)⁻¹ Xᵀ y`

Where:
- `X` is the design matrix, which includes the log distances and a bias term (constant).  
- `y` is the vector of RSS values.


The linear regression model helps establish a baseline and provides a straightforward relationship between log distance and RSS values.

### Neural Network
A deep neural network (DNN) is built using TensorFlow/Keras to predict the RSS values from the log distance. The network architecture consists of:

- An input layer of size 1 (log distance).
- Two hidden layers with ReLU activations (640 and 480 units).
- An output layer with a linear activation function to predict the continuous RSS value.
- The neural network is designed to capture more complex, non-linear relationships between log distance and RSS values.

### Data Preprocessing
The data is normalized using MinMaxScaler to scale the log distance and RSS values to the range [0, 1]. This normalization improves the convergence of the neural network during training. The dataset is then split into training and testing sets, following an 80:20 ratio.

## Usage
Prepare the Data: Ensure you have locations.txt and rss_values.txt files in the directory. These files should contain the coordinates and RSS values, respectively.

Run the Model: Execute the Python script to train both the linear regression model and the neural network model.

```
python model.py
```
Evaluate the Models: The script will evaluate both models on the test set and output the Mean Squared Error (MSE) and Mean Absolute Error (MAE) for both the linear regression and neural network models.

Results: The script will plot:

A graph comparing the training and testing data with the predicted values.
Training and testing loss curves over epochs.
A bar plot showing the performance metrics (MSE and MAE)
