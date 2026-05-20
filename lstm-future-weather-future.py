# ============================================================
# REAL WORLD TEMPERATURE PREDICTION USING LSTM
# ============================================================
#
# DATASET:
# Daily Minimum Temperatures Dataset
#
# Source:
# Automatically downloaded from GitHub
#
# THIS PROJECT:
#
# Real Temperature Data
#          ↓
# Preprocessing
#          ↓
# LSTM Learns Weather Patterns
#          ↓
# Predict Future Temperature
#
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

# NumPy
# Used for numerical computations
import numpy as np

# Pandas
# Used for dataframe handling
import pandas as pd

# Matplotlib
# Used for plotting graphs
import matplotlib.pyplot as plt

# ============================================================
# IMPORT LSTM MODULES
# ============================================================

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    LSTM,
    Dense
)

# ============================================================
# IMPORT PREPROCESSING
# ============================================================

from sklearn.preprocessing import MinMaxScaler

# ============================================================
# IMPORT METRICS
# ============================================================

from sklearn.metrics import (

    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ============================================================
# STEP 1: DOWNLOAD REAL TEMPERATURE DATA
# ============================================================

print("\n========== LOADING REAL TEMPERATURE DATA ==========\n")

# Real-world dataset:
#
# Daily minimum temperatures in Melbourne
#
# Dataset contains:
# - Date
# - Temperature

url = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/"
    "daily-min-temperatures.csv"
)

# Load CSV
data = pd.read_csv(url)

# ============================================================
# DISPLAY DATASET
# ============================================================

print(data.head())

# ============================================================
# STEP 2: EXTRACT TEMPERATURE COLUMN
# ============================================================

# Use only temperature values

temp = data['Temp'].values

# Convert into 2D array
temp = temp.reshape(-1,1)

print("\nTemperature Shape :", temp.shape)

# ============================================================
# STEP 3: NORMALIZATION
# ============================================================

print("\n========== NORMALIZATION ==========\n")

# Scale values between:
#
# 0 → 1
#
# Benefits:
# - stable training
# - faster convergence
# - smoother gradients

scaler = MinMaxScaler()

temp_scaled = scaler.fit_transform(temp)

print("Normalization Completed!")

# ============================================================
# STEP 4: CREATE SEQUENCES
# ============================================================

print("\n========== CREATING SEQUENCES ==========\n")

# Previous 30 days
#        ↓
# Predict next day

X = []
y = []

# Sliding window approach
for i in range(30, len(temp_scaled)):

    # Previous 30 days
    X.append(temp_scaled[i-30:i])

    # Current day target
    y.append(temp_scaled[i])

# Convert into arrays
X = np.array(X)
y = np.array(y)

# ============================================================
# DISPLAY SHAPES
# ============================================================

print("Input Shape  :", X.shape)
print("Target Shape :", y.shape)

# ============================================================
# UNDERSTANDING SHAPE
# ============================================================

# Shape:
#
# (samples, timesteps, features)
#
# Example:
#
# (3620, 30, 1)
#
# Meaning:
#
# 3620 sequences
# 30 previous days
# 1 feature (temperature)

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

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================================
# STEP 6: BUILD LSTM MODEL
# ============================================================

print("\n========== BUILDING LSTM MODEL ==========\n")

# Sequential model
model = Sequential()

# ------------------------------------------------------------
# LSTM LAYER
# ------------------------------------------------------------

# LSTM:
# Long Short-Term Memory
#
# Learns temporal dependencies
#
# 50:
# number of neurons / memory cells

model.add(

    LSTM(

        50,

        input_shape=(X.shape[1],1)
    )
)

# ============================================================
# OUTPUT LAYER
# ============================================================

# Predict one temperature value

model.add(Dense(1))

# ============================================================
# DISPLAY MODEL SUMMARY
# ============================================================

model.summary()

# ============================================================
# STEP 7: COMPILE MODEL
# ============================================================

print("\n========== COMPILING MODEL ==========\n")

# Adam optimizer:
# adaptive optimizer
#
# MSE:
# regression loss function

model.compile(

    optimizer='adam',

    loss='mean_squared_error'
)

print("Model Compilation Completed!")

# ============================================================
# STEP 8: TRAIN MODEL
# ============================================================

print("\n========== TRAINING MODEL ==========\n")

history = model.fit(

    X_train,
    y_train,

    epochs=10,

    batch_size=32,

    validation_data=(X_test, y_test)
)

print("\nModel Training Completed!")

# ============================================================
# STEP 9: MAKE PREDICTIONS
# ============================================================

print("\n========== MAKING PREDICTIONS ==========\n")

# Predict normalized values
predicted_scaled = model.predict(X_test)

# Convert predictions back into real temperatures
predicted = scaler.inverse_transform(predicted_scaled)

# Convert actual values too
actual = scaler.inverse_transform(y_test)

# ============================================================
# STEP 10: EVALUATION
# ============================================================

print("\n========== MODEL EVALUATION ==========\n")

# Mean Squared Error
mse = mean_squared_error(actual, predicted)

# Mean Absolute Error
mae = mean_absolute_error(actual, predicted)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R² Score
r2 = r2_score(actual, predicted)

print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# ============================================================
# STEP 11: DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n========== SAMPLE PREDICTIONS ==========\n")

for i in range(-5, 0):

    actual_val = actual[i][0]

    pred_val = predicted[i][0]

    print(

        f"Actual Temp: {actual_val:.2f}°C | "
        f"Predicted Temp: {pred_val:.2f}°C"
    )

# ============================================================
# STEP 12: VISUALIZATION
# ============================================================

print("\n========== PLOTTING RESULTS ==========\n")

plt.figure(figsize=(12,6))

# Actual temperatures
plt.plot(

    actual,

    label="Actual Temperature"
)

# Predicted temperatures
plt.plot(

    predicted,

    label="Predicted Temperature"
)

# Graph title
plt.title("Real World Temperature Prediction using LSTM")

# Axis labels
plt.xlabel("Days")

plt.ylabel("Temperature")

# Show legend
plt.legend()

# Show graph
plt.show()

# ============================================================
# FINAL FLOW
# ============================================================

# Real Temperature Data
#          ↓
# Normalization
#          ↓
# Create Sequences
#          ↓
# LSTM Learns Weather Patterns
#          ↓
# Predict Future Temperature
#          ↓
# Inverse Scaling
#          ↓
# Evaluation
#          ↓
# Visualization
#
# ============================================================

# ============================================================
# IMPORTANT CONCEPTS USED
# ============================================================

# 1. Time Series
# Sequential data collected over time
#
# 2. LSTM
# Deep learning model for sequence learning
#
# 3. Sliding Window
# Uses previous 30 days to predict next day
#
# 4. Temporal Dependency
# Future depends on past
#
# 5. Normalization
# Scaling values between 0 and 1
#
# 6. Regression
# Predicting continuous values
#
# ============================================================