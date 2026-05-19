# ==========================================
# K-NEAREST NEIGHBOUR (KNN) - IRIS DATASET
# ==========================================

# Import numerical library
import numpy as np

# Import pandas for dataframe handling
import pandas as pd

# Load Iris dataset from sklearn
from sklearn.datasets import load_iris

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Import accuracy metric
from sklearn.metrics import accuracy_score


# ==========================================
# LOAD THE IRIS DATASET
# ==========================================

# load_iris() loads built-in iris dataset
iris = load_iris()

# Features (input data)
# Contains:
# sepal length
# sepal width
# petal length
# petal width

X = iris.data

# Target labels
# 0 -> setosa
# 1 -> versicolor
# 2 -> virginica

y = iris.target


# ==========================================
# DISPLAY DATASET INFORMATION
# ==========================================

# Print feature names
print("Features:", iris.feature_names)

# Print target class names
print("Target names:", iris.target_names)


# ==========================================
# CREATE DATAFRAME (OPTIONAL)
# ==========================================

# Convert dataset into dataframe for better visualization
iris_df = pd.DataFrame(X, columns=iris.feature_names)

# Add target column
iris_df['species'] = y

# Display first 5 rows
print("\nFirst 5 rows of dataset:\n")
print(iris_df.head())


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

# Split dataset into training and testing data
#
# test_size=0.3
# 30% testing data
#
# random_state=42
# ensures same random split every run
#
# stratify=y
# maintains class distribution in train/test sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Print dataset sizes
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# ==========================================
# CREATE KNN MODEL
# ==========================================

# Initialize KNN classifier
#
# n_neighbors=5 means:
# model checks nearest 5 neighbours

knn = KNeighborsClassifier(n_neighbors=5)


# ==========================================
# TRAIN THE MODEL
# ==========================================

# fit() stores training data internally
# KNN is a lazy learner
# It does not create mathematical equation

knn.fit(X_train, y_train)

print("\nKNN model trained successfully.")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

# predict() classifies test samples
y_pred = knn.predict(X_test)


# ==========================================
# EVALUATE MODEL
# ==========================================

# Calculate accuracy
#
# Accuracy =
# correct predictions / total predictions

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy of KNN Model: {accuracy:.2f}")


# ==========================================
# STORE CORRECT & WRONG PREDICTIONS
# ==========================================

correct_predictions = []
wrong_predictions = []


# Loop through all test samples
for i in range(len(y_test)):

    # Check if prediction is correct
    if y_test[i] == y_pred[i]:

        # Store correct prediction details
        correct_predictions.append({

            'Features': X_test[i],

            'Actual': iris.target_names[y_test[i]],

            'Predicted': iris.target_names[y_pred[i]]

        })

    else:

        # Store wrong prediction details
        wrong_predictions.append({

            'Features': X_test[i],

            'Actual': iris.target_names[y_test[i]],

            'Predicted': iris.target_names[y_pred[i]]

        })


# ==========================================
# DISPLAY RESULTS
# ==========================================

print(f"\nTotal Correct Predictions: {len(correct_predictions)}")

print(f"Total Wrong Predictions: {len(wrong_predictions)}")


# ==========================================
# PRINT CORRECT PREDICTIONS
# ==========================================

print("\n========== CORRECT PREDICTIONS ==========")

if correct_predictions:

    for pred in correct_predictions:

        print(
            f"Features: {pred['Features']}, "
            f"Actual: {pred['Actual']}, "
            f"Predicted: {pred['Predicted']}"
        )

else:
    print("No correct predictions found.")


# ==========================================
# PRINT WRONG PREDICTIONS
# ==========================================

print("\n========== WRONG PREDICTIONS ==========")

if wrong_predictions:

    for pred in wrong_predictions:

        print(
            f"Features: {pred['Features']}, "
            f"Actual: {pred['Actual']}, "
            f"Predicted: {pred['Predicted']}"
        )

else:
    print("No wrong predictions found.")