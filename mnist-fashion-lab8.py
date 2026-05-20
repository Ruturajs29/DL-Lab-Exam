# ============================================================
# FASHION MNIST CLASSIFICATION USING CNN
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Fashion MNIST dataset
from tensorflow.keras.datasets import fashion_mnist

# Sequential model
from tensorflow.keras.models import Sequential

# CNN layers
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# One-hot encoding
from tensorflow.keras.utils import to_categorical

# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# ============================================================
# DISPLAY SHAPES
# ============================================================

print("Training Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# ============================================================
# STEP 2: RESHAPE DATA
# ============================================================

# CNN expects:
#
# (samples, height, width, channels)
#
# Fashion MNIST images:
# 28x28 grayscale
#
# So channels = 1

X_train = X_train.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1, 28, 28, 1)

# ============================================================
# STEP 3: NORMALIZATION
# ============================================================

# Convert pixel values:
#
# 0-255 → 0-1

X_train = X_train / 255.0

X_test = X_test / 255.0

# ============================================================
# STEP 4: ONE HOT ENCODING
# ============================================================

# Example:
#
# 3 → [0 0 0 1 0 0 0 0 0 0]

y_train_cat = to_categorical(y_train, 10)

y_test_cat = to_categorical(y_test, 10)

# ============================================================
# STEP 5: BUILD CNN MODEL
# ============================================================

model = Sequential()

# ------------------------------------------------------------
# FIRST CONVOLUTION BLOCK
# ------------------------------------------------------------

model.add(

    Conv2D(

        filters=32,

        kernel_size=(3,3),

        activation='relu',

        input_shape=(28,28,1)
    )
)

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

# ============================================================
# FLATTEN
# ============================================================

model.add(Flatten())

# ============================================================
# DENSE LAYER
# ============================================================

model.add(Dense(128, activation='relu'))

# ============================================================
# DROPOUT
# ============================================================

model.add(Dropout(0.5))

# ============================================================
# OUTPUT LAYER
# ============================================================

# 10 neurons because:
# 10 clothing categories

model.add(Dense(10, activation='softmax'))

# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# ============================================================
# TRAIN MODEL
# ============================================================

history = model.fit(

    X_train,
    y_train_cat,

    epochs=5,

    batch_size=64,

    validation_data=(X_test, y_test_cat)
)

# ============================================================
# MODEL EVALUATION
# ============================================================

loss, accuracy = model.evaluate(X_test, y_test_cat)

print("\nTest Accuracy:", accuracy)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred_probs = model.predict(X_test)

# Convert probabilities into labels
y_pred = np.argmax(y_pred_probs, axis=1)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))

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
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ============================================================
# DISPLAY SAMPLE PREDICTIONS
# ============================================================

class_names = [

    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

for i in range(5):

    plt.imshow(
        X_test[i].reshape(28,28),
        cmap='gray'
    )

    plt.title(

        f"Actual: {class_names[y_test[i]]}\n"
        f"Predicted: {class_names[y_pred[i]]}"
    )

    plt.axis('off')

    plt.show()