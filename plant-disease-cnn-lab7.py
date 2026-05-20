# ============================================================
# PLANT DISEASE DETECTION USING CNN
# ============================================================
#
# THIS CODE:
#
# 1. Downloads CIFAR-10 dataset automatically
# 2. Uses selected classes as plant disease categories
# 3. Preprocesses images
# 4. Builds CNN model
# 5. Trains CNN
# 6. Evaluates model
# 7. Displays confusion matrix
# 8. Shows sample predictions
#
# NOTE:
# ------------------------------------------------------------
# TensorFlow does not provide an inbuilt plant disease dataset
# like MNIST/Fashion-MNIST.
#
# So for a SELF-CONTAINED runnable assignment,
# we use CIFAR-10 images and simulate plant disease classes.
#
# This is useful for:
# - CNN learning
# - academic assignments
# - understanding image classification pipeline
#
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

# NumPy
# Used for arrays and numerical operations
import numpy as np

# Matplotlib
# Used for plotting images and graphs
import matplotlib.pyplot as plt

# Seaborn
# Used for confusion matrix heatmap
import seaborn as sns

# ============================================================
# IMPORT DATASET
# ============================================================

# CIFAR-10 dataset
#
# Automatically downloads dataset
# during first execution
from tensorflow.keras.datasets import cifar10

# ============================================================
# IMPORT CNN MODULES
# ============================================================

# Sequential model:
# layers added one after another
from tensorflow.keras.models import Sequential

# CNN layers
from tensorflow.keras.layers import (

    Conv2D,          # convolution layer
    MaxPooling2D,    # pooling layer
    Flatten,         # converts feature maps → vector
    Dense,           # fully connected layer
    Dropout          # reduces overfitting
)

# One-hot encoding
from tensorflow.keras.utils import to_categorical

# ============================================================
# IMPORT EVALUATION METRICS
# ============================================================

from sklearn.metrics import (

    confusion_matrix,
    classification_report
)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n========== LOADING DATASET ==========\n")

# CIFAR-10 classes:
#
# 0 airplane
# 1 automobile
# 2 bird
# 3 cat
# 4 deer
# 5 dog
# 6 frog
# 7 horse
# 8 ship
# 9 truck
#
# We will simulate plant disease categories:
#
# bird  -> Healthy Leaf
# deer  -> Leaf Rust
# frog  -> Leaf Spot

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# ============================================================
# DISPLAY ORIGINAL SHAPES
# ============================================================

print("Original Training Shape :", X_train.shape)
print("Original Testing Shape  :", X_test.shape)

# ============================================================
# STEP 2: SELECT ONLY 3 CLASSES
# ============================================================

# Selected classes:
#
# 2 -> Healthy Leaf
# 4 -> Leaf Rust
# 6 -> Leaf Spot

selected_classes = [2, 4, 6]

# ============================================================
# CREATE MASKS
# ============================================================

# np.isin() checks whether labels belong to selected classes

train_mask = np.isin(
    y_train,
    selected_classes
).flatten()

test_mask = np.isin(
    y_test,
    selected_classes
).flatten()

# ============================================================
# FILTER DATASET
# ============================================================

X_train = X_train[train_mask]
y_train = y_train[train_mask]

X_test = X_test[test_mask]
y_test = y_test[test_mask]

# ============================================================
# REMAP LABELS
# ============================================================

# Original labels:
#
# 2,4,6
#
# Convert into:
#
# 0,1,2
#
# Required for classification

label_map = {

    2:0,
    4:1,
    6:2
}

# Apply mapping to training labels
y_train = np.array([

    label_map[label[0]]

    for label in y_train
])

# Apply mapping to testing labels
y_test = np.array([

    label_map[label[0]]

    for label in y_test
])

# ============================================================
# DISPLAY NEW SHAPES
# ============================================================

print("\nFiltered Training Shape :", X_train.shape)
print("Filtered Testing Shape  :", X_test.shape)

# ============================================================
# STEP 3: NORMALIZATION
# ============================================================

print("\n========== NORMALIZATION ==========\n")

# Pixel values originally:
#
# 0 → 255
#
# Normalize into:
#
# 0 → 1
#
# Benefits:
# - stable gradients
# - faster convergence
# - smoother optimization

X_train = X_train / 255.0
X_test = X_test / 255.0

print("Normalization Completed!")

# ============================================================
# STEP 4: ONE HOT ENCODING
# ============================================================

print("\n========== ONE HOT ENCODING ==========\n")

# Example:
#
# 0 → [1 0 0]
# 1 → [0 1 0]
# 2 → [0 0 1]

y_train_cat = to_categorical(y_train, 3)
y_test_cat = to_categorical(y_test, 3)

print("One Hot Encoding Completed!")

# ============================================================
# STEP 5: BUILD CNN MODEL
# ============================================================

print("\n========== BUILDING CNN MODEL ==========\n")

# Sequential model
model = Sequential()

# ------------------------------------------------------------
# FIRST CONVOLUTION BLOCK
# ------------------------------------------------------------

# Conv2D:
#
# Learns image features:
# - edges
# - textures
# - patterns
#
# 32 filters:
# model learns 32 feature detectors
#
# (3,3):
# filter size
#
# ReLU:
# activation function

model.add(

    Conv2D(

        filters=32,

        kernel_size=(3,3),

        activation='relu',

        input_shape=(32,32,3)
    )
)

# ------------------------------------------------------------
# MAX POOLING
# ------------------------------------------------------------

# Reduces feature map size
#
# Benefits:
# - less computation
# - less overfitting

model.add(

    MaxPooling2D(pool_size=(2,2))
)

# ------------------------------------------------------------
# SECOND CONVOLUTION BLOCK
# ------------------------------------------------------------

model.add(

    Conv2D(

        filters=64,

        kernel_size=(3,3),

        activation='relu'
    )
)

model.add(

    MaxPooling2D(pool_size=(2,2))
)

# ------------------------------------------------------------
# THIRD CONVOLUTION BLOCK
# ------------------------------------------------------------

model.add(

    Conv2D(

        filters=128,

        kernel_size=(3,3),

        activation='relu'
    )
)

model.add(

    MaxPooling2D(pool_size=(2,2))
)

# ============================================================
# FLATTEN LAYER
# ============================================================

# Converts multidimensional feature maps
# into single vector

model.add(Flatten())

# ============================================================
# DENSE LAYER
# ============================================================

# Fully connected layer

model.add(

    Dense(

        128,

        activation='relu'
    )
)

# ============================================================
# DROPOUT LAYER
# ============================================================

# Randomly disables neurons during training
#
# Helps reduce overfitting

model.add(

    Dropout(0.5)
)

# ============================================================
# OUTPUT LAYER
# ============================================================

# 3 neurons:
#
# Healthy Leaf
# Leaf Rust
# Leaf Spot
#
# Softmax:
# converts outputs into probabilities

model.add(

    Dense(

        3,

        activation='softmax'
    )
)

# ============================================================
# DISPLAY MODEL SUMMARY
# ============================================================

model.summary()

# ============================================================
# STEP 6: COMPILE MODEL
# ============================================================

print("\n========== COMPILING MODEL ==========\n")

# optimizer='adam'
#
# Adam optimizer:
# efficient weight updates
#
# loss='categorical_crossentropy'
#
# Used for multiclass classification

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

print("Model Compilation Completed!")

# ============================================================
# STEP 7: TRAIN MODEL
# ============================================================

print("\n========== TRAINING MODEL ==========\n")

# epochs=5
#
# Entire dataset passed 5 times
#
# batch_size=64
#
# Processes 64 images together

history = model.fit(

    X_train,
    y_train_cat,

    epochs=5,

    batch_size=64,

    validation_data=(X_test, y_test_cat)
)

print("\nModel Training Completed!")

# ============================================================
# STEP 8: EVALUATE MODEL
# ============================================================

print("\n========== MODEL EVALUATION ==========\n")

loss, accuracy = model.evaluate(

    X_test,
    y_test_cat
)

print("\nTest Accuracy :", accuracy)

# ============================================================
# STEP 9: PREDICTIONS
# ============================================================

# Predict probabilities
y_pred_probs = model.predict(X_test)

# Convert probabilities → labels
y_pred = np.argmax(y_pred_probs, axis=1)

# ============================================================
# STEP 10: CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues'
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# STEP 11: CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(classification_report(y_test, y_pred))

# ============================================================
# STEP 12: DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n========== SAMPLE PREDICTIONS ==========\n")

class_names = [

    "Healthy Leaf",
    "Leaf Rust",
    "Leaf Spot"
]

# Display first 5 predictions
for i in range(5):

    plt.imshow(X_test[i])

    plt.title(

        f"Actual: {class_names[y_test[i]]}\n"
        f"Predicted: {class_names[y_pred[i]]}"
    )

    plt.axis('off')

    plt.show()

# ============================================================
# STEP 13: PLOT TRAINING ACCURACY
# ============================================================

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.show()

# ============================================================
# FINAL FLOW OF PROJECT
# ============================================================

# Dataset
#    ↓
# Preprocessing
#    ↓
# CNN Feature Extraction
#    ↓
# Pooling
#    ↓
# Flatten
#    ↓
# Dense Layers
#    ↓
# Softmax Classification
#    ↓
# Plant Disease Prediction
#
# ============================================================