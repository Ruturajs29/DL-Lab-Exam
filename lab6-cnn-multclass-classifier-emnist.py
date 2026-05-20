# ============================================================
# ASSIGNMENT 6
# CNN Multiclass Classifier using MNIST Dataset
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

# Numerical computations
import numpy as np

# Plotting graphs and images
import matplotlib.pyplot as plt

# Better visualization for confusion matrix
import seaborn as sns

# MNIST dataset (handwritten digits)
from tensorflow.keras.datasets import mnist

# Sequential model = layers stacked one after another
from tensorflow.keras.models import Sequential

# CNN layers
from tensorflow.keras.layers import (
    Conv2D,          # Convolution layer
    MaxPooling2D,    # Pooling layer
    Flatten,         # Converts 2D → 1D
    Dense            # Fully connected layer
)

# Converts labels into one-hot encoded format
from tensorflow.keras.utils import to_categorical

# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# PART (a) DATA PREPROCESSING
# ============================================================

print("\n========== DATA PREPROCESSING ==========\n")

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
# MNIST contains handwritten digit images (0–9)
#
# X_train -> training images
# y_train -> training labels
# X_test  -> testing images
# y_test  -> testing labels
#
# Shape of X_train:
# (60000, 28, 28)
#
# Meaning:
# 60000 images
# each image size = 28x28 pixels
# ------------------------------------------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ------------------------------------------------------------
# DISPLAY SHAPES
# ------------------------------------------------------------

print("Training Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)

# ------------------------------------------------------------
# RESHAPE IMAGES
# ------------------------------------------------------------
# CNN expects input in this format:
#
# (samples, height, width, channels)
#
# Since MNIST is grayscale:
# channels = 1
#
# Original:
# (60000, 28, 28)
#
# After reshape:
# (60000, 28, 28, 1)
# ------------------------------------------------------------

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------
# Pixel values originally range from:
# 0 → 255
#
# Neural networks train better when values are small.
#
# So we scale values into:
# 0 → 1
#
# Formula:
# normalized_value = pixel / 255
# ------------------------------------------------------------

X_train = X_train / 255.0
X_test = X_test / 255.0

# ------------------------------------------------------------
# ONE HOT ENCODING
# ------------------------------------------------------------
# Labels are originally:
# 5, 0, 7, 2 ...
#
# Convert into:
#
# 5 → [0 0 0 0 0 1 0 0 0 0]
#
# Because output layer has 10 neurons.
# ------------------------------------------------------------

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("\nData Preprocessing Completed!\n")

# ============================================================
# PART (b) BUILD CNN MODEL
# ============================================================

print("\n========== BUILDING CNN MODEL ==========\n")

# ------------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------------
# Sequential means:
# layers are added one after another.
# ------------------------------------------------------------

model = Sequential([

    # ========================================================
    # FIRST CONVOLUTION LAYER
    # ========================================================

    # Conv2D:
    # Applies filters/kernels on image
    #
    # 32 = number of filters
    #
    # (3,3) = filter size
    #
    # activation='relu'
    # ReLU helps introduce non-linearity
    #
    # input_shape=(28,28,1)
    # height=28
    # width=28
    # channels=1 (grayscale)
    #
    # This layer learns:
    # edges, lines, corners
    # ========================================================

    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    ),

    # ========================================================
    # MAX POOLING LAYER
    # ========================================================
    # Reduces image size
    #
    # (2,2) means:
    # take maximum from every 2x2 region
    #
    # Benefits:
    # - reduces computation
    # - reduces overfitting
    # - keeps important features
    # ========================================================

    MaxPooling2D(pool_size=(2, 2)),

    # ========================================================
    # SECOND CONVOLUTION LAYER
    # ========================================================
    # More filters = learning more complex features
    #
    # Earlier layer:
    # learns edges
    #
    # Later layer:
    # learns digit shapes/patterns
    # ========================================================

    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ),

    # ========================================================
    # SECOND POOLING LAYER
    # ========================================================

    MaxPooling2D(pool_size=(2, 2)),

    # ========================================================
    # FLATTEN LAYER
    # ========================================================
    # Converts multidimensional data into vector
    #
    # Example:
    # 5x5x64 → 1600 vector
    #
    # Needed before Dense layers
    # ========================================================

    Flatten(),

    # ========================================================
    # DENSE (FULLY CONNECTED) LAYER
    # ========================================================
    # 128 neurons
    #
    # Learns high-level combinations of features
    # ========================================================

    Dense(128, activation='relu'),

    # ========================================================
    # OUTPUT LAYER
    # ========================================================
    # 10 neurons because:
    # digits = 0 to 9
    #
    # Softmax converts outputs into probabilities
    #
    # Example output:
    # [0.01,0.02,0.90,...]
    # ========================================================

    Dense(10, activation='softmax')
])

# ============================================================
# MODEL SUMMARY
# ============================================================
# Shows:
# - layers
# - output shapes
# - number of parameters
# ============================================================

model.summary()

# ============================================================
# COMPILE MODEL
# ============================================================

# optimizer='adam'
# Adam optimizer updates weights efficiently
#
# loss='categorical_crossentropy'
# Used for multiclass classification
#
# metrics=['accuracy']
# Tracks accuracy during training
# ============================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n========== TRAINING MODEL ==========\n")

# model.fit() starts learning
#
# epochs=5
# means complete dataset is passed 5 times
#
# batch_size=64
# model processes 64 images at once
#
# validation_data
# checks performance on unseen data
# ============================================================

history = model.fit(
    X_train,
    y_train_cat,

    epochs=5,
    batch_size=64,

    validation_data=(X_test, y_test_cat)
)

print("\nModel Training Completed!\n")

# ============================================================
# PART (c) MODEL EVALUATION
# ============================================================

print("\n========== MODEL EVALUATION ==========\n")

# ------------------------------------------------------------
# PREDICT PROBABILITIES
# ------------------------------------------------------------
# Example output:
#
# [0.01, 0.02, 0.91, ...]
#
# Highest probability = predicted class
# ------------------------------------------------------------

y_pred_probs = model.predict(X_test)

# ------------------------------------------------------------
# CONVERT PROBABILITIES → CLASS LABELS
# ------------------------------------------------------------
# np.argmax() returns index of highest probability
#
# Example:
# [0.01,0.02,0.91]
#
# Prediction = class 2
# ------------------------------------------------------------

y_pred = np.argmax(y_pred_probs, axis=1)

# ============================================================
# CONFUSION MATRIX
# ============================================================
# Shows:
# actual vs predicted classes
#
# Diagonal values:
# correct predictions
#
# Off-diagonal values:
# wrong predictions
# ============================================================

cm = confusion_matrix(y_test, y_pred)

# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,      # show numbers
    fmt='d',         # integer format
    cmap='Blues'     # color theme
)

plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")

plt.show()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
# Contains:
#
# Precision
# Recall
# F1-score
# Accuracy
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(classification_report(y_test, y_pred))

# ============================================================
# FINAL TEST ACCURACY
# ============================================================

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)

print("\nFinal Test Accuracy :", test_accuracy)

# ============================================================
# OPTIONAL: DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n========== SAMPLE PREDICTIONS ==========\n")

# Display first 5 test images with predictions

for i in range(5):

    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')

    plt.title(
        f"Actual: {y_test[i]} | Predicted: {y_pred[i]}"
    )

    plt.axis('off')

    plt.show()