# ============================================================
# ASSIGNMENT:
# STOCK PRICE PREDICTION USING RNN
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

# NumPy
# Used for numerical computations and arrays
import numpy as np

# Pandas
# Used for handling tabular/time-series data
import pandas as pd

# Matplotlib
# Used for plotting graphs
import matplotlib.pyplot as plt

# yfinance
# Used to download stock market data from Yahoo Finance
import yfinance as yf

# ============================================================
# IMPORT DEEP LEARNING MODULES
# ============================================================

# Sequential model:
# layers are added one after another
from tensorflow.keras.models import Sequential

# SimpleRNN:
# basic recurrent neural network layer
#
# Dense:
# fully connected output layer
from tensorflow.keras.layers import SimpleRNN, Dense

# ============================================================
# IMPORT PREPROCESSING & METRICS
# ============================================================

# MinMaxScaler:
# used for normalization (0 to 1 scaling)
from sklearn.preprocessing import MinMaxScaler

# Regression evaluation metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# ============================================================
# STEP 1: DOWNLOAD STOCK DATA
# ============================================================

print("\n========== DOWNLOADING STOCK DATA ==========\n")

# Download Apple stock data
#
# start -> starting date
# end   -> ending date
#
# Dataset contains:
# Open, High, Low, Close, Volume etc.
data = yf.download(
    'AAPL',
    start='2020-01-01',
    end='2024-01-01'
)

# Display first 5 rows
print(data.head())

# ============================================================
# STEP 2: USE CLOSING PRICE
# ============================================================

# We use only closing price for prediction
#
# Closing price is the final market price of the day
prices = data['Close'].values

# reshape(-1,1)
#
# Converts:
# [100,101,102]
#
# into:
# [[100],
#  [101],
#  [102]]
#
# Required because scaler expects 2D input
prices = prices.reshape(-1, 1)

print("\nClosing Price Shape:", prices.shape)

# ============================================================
# STEP 3: NORMALIZATION
# ============================================================

print("\n========== NORMALIZATION ==========\n")

# MinMaxScaler scales data between:
# 0 and 1
#
# Formula:
#
# x' = (x - xmin) / (xmax - xmin)
#
# Why normalization?
#
# - improves training stability
# - faster convergence
# - prevents large gradients
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit scaler and transform prices
prices_scaled = scaler.fit_transform(prices)

print("Normalization Completed!")

# ============================================================
# STEP 4: CREATE SEQUENCES
# ============================================================

print("\n========== CREATING SEQUENCES ==========\n")

# RNN learns from previous time steps
#
# Here:
# previous 60 days → predict next day
#
# Example:
#
# Day1 → Day60  => predict Day61
# Day2 → Day61  => predict Day62
#
# This is called Sliding Window approach

X = []   # input sequences
y = []   # target values

# Start from day 60
for i in range(60, len(prices_scaled)):

    # Previous 60 days
    X.append(prices_scaled[i-60:i])

    # Current day (target)
    y.append(prices_scaled[i])

# Convert lists into numpy arrays
X = np.array(X)
y = np.array(y)

# Display shapes
print("Input Shape  :", X.shape)
print("Target Shape :", y.shape)

# ============================================================
# UNDERSTANDING SHAPES
# ============================================================

# X shape:
#
# (samples, timesteps, features)
#
# Example:
#
# (946, 60, 1)
#
# Meaning:
#
# 946 sequences
# each sequence has 60 timesteps
# each timestep has 1 feature (closing price)

# ============================================================
# STORE TARGET DATES
# ============================================================

# Dates corresponding to target values
target_dates = data.index[60:]

# ============================================================
# STEP 5: TRAIN TEST SPLIT
# ============================================================

print("\n========== TRAIN TEST SPLIT ==========\n")

# 80% training
# 20% testing

split_idx = int(len(X) * 0.8)

# Training data
X_train = X[:split_idx]
y_train = y[:split_idx]

# Testing data
X_test = X[split_idx:]
y_test = y[split_idx:]

# Store test dates
test_dates = target_dates[split_idx:]

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================================
# IMPORTANT:
# ============================================================

# In time series:
# NEVER randomly shuffle data
#
# Because:
# future should not come before past
#
# Must preserve chronological order

# ============================================================
# STEP 6: BUILD RNN MODEL
# ============================================================

print("\n========== BUILDING RNN MODEL ==========\n")

# Sequential model:
# layers added one after another
model = Sequential()

# ------------------------------------------------------------
# SIMPLE RNN LAYER
# ------------------------------------------------------------

# SimpleRNN:
# recurrent neural network layer
#
# 50 = number of neurons / hidden units
#
# input_shape=(60,1)
#
# 60 -> timesteps (previous 60 days)
# 1  -> feature count
#
# RNN remembers previous information using hidden state

model.add(
    SimpleRNN(
        units=50,
        input_shape=(X.shape[1], 1)
    )
)

# ------------------------------------------------------------
# DENSE OUTPUT LAYER
# ------------------------------------------------------------

# Dense(1):
# one neuron because:
# predicting one stock price

model.add(Dense(1))

# ============================================================
# MODEL SUMMARY
# ============================================================

# Shows:
# layers
# output shapes
# parameters
model.summary()

# ============================================================
# STEP 7: COMPILE MODEL
# ============================================================

print("\n========== COMPILING MODEL ==========\n")

# optimizer='adam'
#
# Adam optimizer updates weights efficiently
#
# loss='mean_squared_error'
#
# Used for regression problems
#
# MSE Formula:
#
# MSE = average((actual - predicted)^2)

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

print("Model Compilation Completed!")

# ============================================================
# STEP 8: TRAIN MODEL
# ============================================================

print("\n========== TRAINING MODEL ==========\n")

# model.fit() starts training
#
# epochs=5
# entire dataset passes 5 times
#
# batch_size=32
# processes 32 sequences together

history = model.fit(
    X_train,
    y_train,

    epochs=5,
    batch_size=32
)

print("\nModel Training Completed!")

# ============================================================
# STEP 9: PREDICT STOCK PRICES
# ============================================================

print("\n========== MAKING PREDICTIONS ==========\n")

# Predict normalized values
predicted_scaled = model.predict(X_test)

# ============================================================
# INVERSE TRANSFORM
# ============================================================

# Model predictions are normalized
#
# Convert back to actual stock prices

predicted = scaler.inverse_transform(predicted_scaled)

# Convert actual test values back too
actual = scaler.inverse_transform(y_test)

# ============================================================
# STEP 10: EVALUATION METRICS
# ============================================================

print("\n========== EVALUATION METRICS ==========\n")

# ------------------------------------------------------------
# MEAN SQUARED ERROR (MSE)
# ------------------------------------------------------------
#
# Measures average squared error
#
# Lower = better
#
# Formula:
#
# MSE = average((actual - predicted)^2)

mse = mean_squared_error(actual, predicted)

# ------------------------------------------------------------
# MEAN ABSOLUTE ERROR (MAE)
# ------------------------------------------------------------
#
# Average absolute difference
#
# Easier to interpret than MSE

mae = mean_absolute_error(actual, predicted)

# ------------------------------------------------------------
# ROOT MEAN SQUARED ERROR (RMSE)
# ------------------------------------------------------------
#
# Square root of MSE
#
# Gives error in original units

rmse = np.sqrt(mse)

# ------------------------------------------------------------
# R² SCORE
# ------------------------------------------------------------
#
# Measures goodness of fit
#
# 1  -> perfect
# 0  -> poor
# <0 -> terrible

r2 = r2_score(actual, predicted)

# ------------------------------------------------------------
# MAPE
# ------------------------------------------------------------
#
# Mean Absolute Percentage Error
#
# Gives percentage error

mape = mean_absolute_percentage_error(
    actual,
    predicted
) * 100.0

# ============================================================
# DISPLAY METRICS
# ============================================================

print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")

# ============================================================
# STEP 11: DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n========== SAMPLE PREDICTIONS ==========\n")

# Display last 5 predictions

for i in range(-5, 0):

    # Convert date into string
    date_str = test_dates[i].strftime("%Y-%m-%d")

    # Actual stock price
    actual_val = actual[i][0]

    # Predicted stock price
    pred_val = predicted[i][0]

    print(
        f"{date_str} | "
        f"Actual: {actual_val:.2f} | "
        f"Predicted: {pred_val:.2f}"
    )

# ============================================================
# STEP 12: VISUALIZATION
# ============================================================

print("\n========== PLOTTING RESULTS ==========\n")

# Plot actual stock prices
plt.plot(
    test_dates,
    actual,
    label="Actual Price"
)

# Plot predicted stock prices
plt.plot(
    test_dates,
    predicted,
    label="Predicted Price"
)

# Graph labels
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Apple Stock Price Prediction using RNN")

# Show legend
plt.legend()

# Display graph
plt.show()

# ============================================================
# FINAL UNDERSTANDING
# ============================================================

# FLOW OF ENTIRE PROJECT:
#
# Historical Prices
#         ↓
# Normalization
#         ↓
# Create 60-day Sequences
#         ↓
# RNN Learns Temporal Patterns
#         ↓
# Predict Next Price
#         ↓
# Inverse Scaling
#         ↓
# Evaluate Metrics
#         ↓
# Visualize Predictions
#
# ============================================================