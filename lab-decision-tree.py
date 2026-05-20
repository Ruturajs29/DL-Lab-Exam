# =========================================================
# ASSIGNMENT 5: DECISION TREES & ENSEMBLE METHODS
# =========================================================

# Import numerical operations library
import numpy as np

# Import pandas for dataframe handling
import pandas as pd

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Import datasets
from sklearn.datasets import load_iris, fetch_california_housing

# Split dataset into train and test sets
from sklearn.model_selection import train_test_split

# Decision Tree models and visualization
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree
)

# Evaluation metrics
from sklearn.metrics import accuracy_score, r2_score

# Ensemble methods
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier
)

# =========================================================
# PART 1: CLASSIFICATION USING IRIS DATASET
# =========================================================

print("\n========== CLASSIFICATION ==========\n")


# =========================================================
# LOAD IRIS DATASET
# =========================================================

# load_iris() loads built-in iris flower dataset
iris = load_iris()

# Input features
# Contains:
# sepal length
# sepal width
# petal length
# petal width

X = iris.data

# Target labels
# 0 -> Setosa
# 1 -> Versicolor
# 2 -> Virginica

y = iris.target


# =========================================================
# SPLIT DATASET
# =========================================================

# Split data into:
# training data -> model learns
# testing data -> model evaluation
#
# test_size=0.2
# 20% data used for testing
#
# random_state=42
# ensures same random split every run

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# DECISION TREE CLASSIFIER
# =========================================================

# Create Decision Tree classifier
dt_clf = DecisionTreeClassifier(random_state=42)

# Train the model
# fit() builds the decision tree
dt_clf.fit(X_train, y_train)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

# Predict training data
y_train_pred = dt_clf.predict(X_train)

# Predict testing data
y_test_pred = dt_clf.predict(X_test)

# Calculate training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

# Display accuracies
print("Decision Tree Train Accuracy:", train_accuracy)
print("Decision Tree Test Accuracy:", test_accuracy)


# =========================================================
# COST COMPLEXITY PRUNING
# =========================================================

# Decision trees can overfit.
# Pruning removes unnecessary branches.

# Get pruning path
path = dt_clf.cost_complexity_pruning_path(X_train, y_train)

# Extract alpha values
ccp_alphas = path.ccp_alphas

# Store pruned models
models = []


# Train multiple pruned trees
for alpha in ccp_alphas:

    # ccp_alpha controls pruning strength
    clf = DecisionTreeClassifier(ccp_alpha=alpha)

    # Train pruned tree
    clf.fit(X_train, y_train)

    # Store model
    models.append(clf)


# =========================================================
# EVALUATE PRUNED MODELS
# =========================================================

# Training accuracies
train_scores = [

    accuracy_score(y_train, model.predict(X_train))

    for model in models
]

# Testing accuracies
test_scores = [

    accuracy_score(y_test, model.predict(X_test))

    for model in models
]

# Select best model
best_model = models[np.argmax(test_scores)]

# Print best accuracy
print("Best Accuracy after Pruning:", max(test_scores))


# =========================================================
# RANDOM FOREST CLASSIFIER
# =========================================================

# Random Forest combines multiple decision trees
# to reduce overfitting

rf_clf = RandomForestClassifier(
    n_estimators=100,   # Number of trees
    random_state=42
)

# Train Random Forest
rf_clf.fit(X_train, y_train)

# Predict testing data
rf_pred = rf_clf.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)


# =========================================================
# ADABOOST CLASSIFIER (DECISION STUMPS)
# =========================================================

# Decision stump:
# Decision tree with max_depth=1

stump = DecisionTreeClassifier(max_depth=1)

# AdaBoost combines multiple weak learners
ada = AdaBoostClassifier(

    estimator=stump,

    n_estimators=50,

    random_state=42
)

# Train AdaBoost model
ada.fit(X_train, y_train)

# Predict testing data
ada_pred = ada.predict(X_test)

# Calculate accuracy
ada_accuracy = accuracy_score(y_test, ada_pred)

print("AdaBoost Accuracy:", ada_accuracy)


# =========================================================
# PART 2: REGRESSION USING CALIFORNIA HOUSING
# =========================================================

print("\n========== REGRESSION ==========\n")


# =========================================================
# LOAD CALIFORNIA HOUSING DATASET
# =========================================================

# fetch_california_housing()
# loads housing price dataset

housing = fetch_california_housing()

# Input features
X = housing.data

# Target values (house prices)
y = housing.target


# =========================================================
# SPLIT DATASET
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# DECISION TREE REGRESSOR
# =========================================================

# Create regression tree
dt_reg = DecisionTreeRegressor(random_state=42)

# Train model
dt_reg.fit(X_train, y_train)


# =========================================================
# REGRESSION PERFORMANCE
# =========================================================

# Predict training data
y_train_pred = dt_reg.predict(X_train)

# Predict testing data
y_test_pred = dt_reg.predict(X_test)

# Calculate R² score for training data
train_r2 = r2_score(y_train, y_train_pred)

# Calculate R² score for testing data
test_r2 = r2_score(y_test, y_test_pred)

# Display results
print("Decision Tree Train R²:", train_r2)
print("Decision Tree Test R²:", test_r2)


# =========================================================
# COST COMPLEXITY PRUNING FOR REGRESSION
# =========================================================

# Get pruning path
path = dt_reg.cost_complexity_pruning_path(X_train, y_train)

# Extract alpha values
ccp_alphas = path.ccp_alphas

# Store models
models = []


# Train multiple pruned regression trees
for alpha in ccp_alphas:

    # Create pruned regressor
    reg = DecisionTreeRegressor(ccp_alpha=alpha)

    # Train model
    reg.fit(X_train, y_train)

    # Store model
    models.append(reg)


# =========================================================
# EVALUATE PRUNED REGRESSION MODELS
# =========================================================

# Training R² scores
train_scores = [

    r2_score(y_train, model.predict(X_train))

    for model in models
]

# Testing R² scores
test_scores = [

    r2_score(y_test, model.predict(X_test))

    for model in models
]

# Best model selection
best_model = models[np.argmax(test_scores)]

# Display best R² score
print("Best R² after Pruning:", max(test_scores))


# =========================================================
# RANDOM FOREST REGRESSOR
# =========================================================

# Random Forest Regressor
rf_reg = RandomForestRegressor(

    n_estimators=100,

    random_state=42
)

# Train model
rf_reg.fit(X_train, y_train)

# Predict testing data
rf_pred = rf_reg.predict(X_test)

# Calculate R² score
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest R²:", rf_r2)


# =========================================================
# TREE VISUALIZATION
# =========================================================

# Create figure
plt.figure(figsize=(10, 6))

# Draw decision tree
plot_tree(

    dt_clf,

    filled=True,

    feature_names=iris.feature_names,

    class_names=iris.target_names
)

# Graph title
plt.title("Decision Tree Classifier")

# Display graph
plt.show()