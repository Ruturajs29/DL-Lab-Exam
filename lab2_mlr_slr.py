# ==========================================
# SIMPLE & MULTIPLE LINEAR REGRESSION
# ==========================================

# Import numerical computation library
import numpy as np

# Import pandas for dataframe handling
import pandas as pd

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Splits dataset into training and testing data
from sklearn.model_selection import train_test_split

# Linear Regression model
from sklearn.linear_model import LinearRegression

# Used for feature scaling / standardization
from sklearn.preprocessing import StandardScaler

# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# CREATE DATASET
# ==========================================

# Creating a dictionary containing house data
data = {
    'Size': [800, 1000, 1200, 1500, 1800, 2000, 2200, None],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 5, 3],
    'Age': [20, 15, 10, 8, 5, 3, 2, 12],
    'Price': [50, 60, 75, 90, 120, 150, 170, 80]
}

# Convert dictionary into pandas dataframe
df = pd.DataFrame(data)

# Display dataset
print("Original Dataset:\n")
print(df)


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

# fillna() replaces missing values (NaN)
# df.mean() calculates mean of each column
# inplace=True modifies original dataframe directly

df.fillna(df.mean(), inplace=True)

print("\nDataset After Handling Missing Values:\n")
print(df)


# ==========================================
# FEATURE SCALING
# ==========================================

# Create scaler object
scaler = StandardScaler()

# List of input features
features = ['Size', 'Bedrooms', 'Age']

# fit_transform():
# fit() -> learns mean and standard deviation
# transform() -> applies standardization formula
#
# Formula:
# z = (x - mean) / standard deviation

df[features] = scaler.fit_transform(df[features])

print("\nDataset After Feature Scaling:\n")
print(df)


# =========================================================
# SIMPLE LINEAR REGRESSION
# Predict Price using only Size
# =========================================================

# Independent variable (input feature)
# Double brackets return dataframe (2D)
X_simple = df[['Size']]

# Dependent variable (target/output)
y = df['Price']


# Split dataset into training and testing data
# test_size=0.2 means 20% data for testing
# random_state=42 ensures same random split every run

X_train, X_test, y_train, y_test = train_test_split(
    X_simple,
    y,
    test_size=0.2,
    random_state=42
)


# Create Linear Regression model
simple_model = LinearRegression()


# Train model using training data
# fit() learns slope and intercept of best fit line
simple_model.fit(X_train, y_train)


# Predict house prices for test data
y_pred_simple = simple_model.predict(X_test)


# ==========================================
# EVALUATE SIMPLE LINEAR REGRESSION
# ==========================================

# Mean Squared Error
# Measures average squared prediction error
mse_simple = mean_squared_error(y_test, y_pred_simple)

# Root Mean Squared Error
# Square root of MSE
rmse_simple = np.sqrt(mse_simple)

# R² Score
# Measures goodness of fit
# Closer to 1 means better model
r2_simple = r2_score(y_test, y_pred_simple)


# Print evaluation metrics
print("\n========== SIMPLE LINEAR REGRESSION ==========")
print("MSE :", mse_simple)
print("RMSE :", rmse_simple)
print("R² Score :", r2_simple)


# ==========================================
# VISUALIZATION OF SIMPLE REGRESSION
# ==========================================

# Scatter plot of actual data points
plt.scatter(X_simple, y, color='blue')

# Regression line
# predict(X_simple) predicts values for all points
plt.plot(X_simple, simple_model.predict(X_simple), color='red')

# Axis labels
plt.xlabel("House Size (Scaled)")
plt.ylabel("House Price")

# Graph title
plt.title("Simple Linear Regression")

# Display graph
plt.show()


# =========================================================
# MULTIPLE LINEAR REGRESSION
# Predict Price using multiple features
# =========================================================

# Multiple input features
X_multi = df[['Size', 'Bedrooms', 'Age']]

# Target variable
y = df['Price']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_multi,
    y,
    test_size=0.2,
    random_state=42
)


# Create Multiple Linear Regression model
multi_model = LinearRegression()


# Train model
multi_model.fit(X_train, y_train)


# Predict test data
y_pred_multi = multi_model.predict(X_test)


# ==========================================
# EVALUATE MULTIPLE LINEAR REGRESSION
# ==========================================

# Mean Squared Error
mse_multi = mean_squared_error(y_test, y_pred_multi)

# Root Mean Squared Error
rmse_multi = np.sqrt(mse_multi)

# R² Score
r2_multi = r2_score(y_test, y_pred_multi)


# Print evaluation metrics
print("\n========== MULTIPLE LINEAR REGRESSION ==========")
print("MSE :", mse_multi)
print("RMSE :", rmse_multi)
print("R² Score :", r2_multi)


# ==========================================
# VISUALIZATION OF PREDICTIONS
# ==========================================

# Scatter plot:
# x-axis -> Actual Prices
# y-axis -> Predicted Prices

plt.scatter(y_test, y_pred_multi, color='green')

# Axis labels
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

# Graph title
plt.title("Multiple Linear Regression Predictions")

# Display graph
plt.show()