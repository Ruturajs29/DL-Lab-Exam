# ==========================================
# NAÏVE BAYES CLASSIFIER - IRIS DATASET
# ==========================================

# Import Iris dataset
from sklearn.datasets import load_iris

# Split dataset into train and test sets
from sklearn.model_selection import train_test_split

# Gaussian Naïve Bayes classifier
from sklearn.naive_bayes import GaussianNB

# Accuracy evaluation metric
from sklearn.metrics import accuracy_score


# ==========================================
# LOAD THE IRIS DATASET
# ==========================================

# load_iris() loads built-in iris flower dataset
iris = load_iris()

# Features (input data)
#
# Contains:
# sepal length
# sepal width
# petal length
# petal width

X = iris.data

# Target labels
#
# 0 -> setosa
# 1 -> versicolor
# 2 -> virginica

y = iris.target


# ==========================================
# SPLIT DATASET
# ==========================================

# Split dataset into:
# training data -> model learns
# testing data -> model evaluation
#
# test_size=0.3
# 30% data used for testing
#
# random_state=42
# ensures same random split every run

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ==========================================
# CREATE NAÏVE BAYES MODEL
# ==========================================

# GaussianNB()
#
# Gaussian Naïve Bayes assumes:
# feature values follow normal distribution

model = GaussianNB()


# ==========================================
# TRAIN THE MODEL
# ==========================================

# fit() trains the model
#
# Model learns:
# - probability of each class
# - mean of each feature
# - variance of each feature

model.fit(X_train, y_train)


# ==========================================
# MAKE PREDICTIONS
# ==========================================

# predict() predicts flower class
# for test data

y_pred = model.predict(X_test)


# ==========================================
# CALCULATE ACCURACY
# ==========================================

# accuracy_score()
#
# Accuracy =
# correct predictions / total predictions

accuracy = accuracy_score(y_test, y_pred)


# ==========================================
# DISPLAY RESULT
# ==========================================

print(f"Naïve Bayes Classifier Accuracy: {accuracy * 100:.2f}%")