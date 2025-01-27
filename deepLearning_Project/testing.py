import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Define the function to calculate log distance
def calculate_log_distance(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return -10 * np.log10(distance)

# Define the function to calculate linear regression weights
def log_distance_regression(X, Y):
    X_transpose = np.transpose(X)
    X_X_transpose = X_transpose @ X
    X_X_transpose_inv = np.linalg.inv(X_X_transpose)
    w = np.matmul(np.matmul(X_X_transpose_inv, X_transpose), Y)
    return w

# Read data from files
X = pd.read_csv('locations.txt', header=None, names=['X', 'Y'], delimiter='\t')
y_original = pd.read_csv('rss_values.txt', header=None, delimiter=' ')

# Calculate log distance matrix
log_distance = np.zeros((len(y_original), len(X)))
for i in range(len(y_original)):
    for j in range(len(X)):
        if y_original.iloc[i, j] == np.inf:
            for k in range(len(X)):
                log_distance[i][k] = calculate_log_distance(X.iloc[j]['X'], X.iloc[j]['Y'], X.iloc[k]['X'], X.iloc[k]['Y'])

# Prepare data for training
x = []
y = []
for i in range(len(y_original)):
    for j in range(len(X)):
        if log_distance[i][j] != np.inf and y_original.iloc[i, j] != np.inf:
            x.append(log_distance[i][j])
            y.append(y_original.iloc[i, j])

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

# Load the saved model
model = tf.keras.models.load_model('final.keras')

# Predict and evaluate the neural network model
y_pred_nn = model.predict(x)
test_loss_nn, test_mae_nn = model.evaluate(x, y, verbose=0)

print("Neural Network Test Loss:", test_loss_nn)

# Perform linear regression
x_lr = np.c_[np.ones(x.shape[0]), x]
w = log_distance_regression(x_lr, y)

# Predict using the linear regression model
y_pred_lr = x_lr @ w

# Calculate loss for linear regression (Mean Squared Error)
loss_lr = mean_squared_error(y, y_pred_lr)
print("Linear Regression Test Loss (MSE):", loss_lr)

# De-normalize the predictions and test labels for visualization
y_original_values = scaler_y.inverse_transform(y)
y_pred_nn_original = scaler_y.inverse_transform(y_pred_nn)
y_pred_lr_original = scaler_y.inverse_transform(y_pred_lr)
x_original = scaler_x.inverse_transform(x)

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot of data
plt.scatter(x_original, y_original_values, label='Data', color='#B0E0E6', marker='o', s=65, edgecolor='#696969')

# Scatter plot of neural network predictions
plt.scatter(x_original, y_pred_nn_original, label='NN Predictions', color='r', marker='x', s=50)

# Linear regression line
plt.plot(x_original, y_pred_lr_original, label='Linear Regression', color='g', linewidth=2)

# Plot aesthetics
plt.xlabel('Log Distance')
plt.ylabel('RSS Value')
plt.title('Data, Neural Network Predictions, and Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
