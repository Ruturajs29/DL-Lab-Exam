# This code performs:

# Load heart dataset
# Convert categorical columns into machine-readable format
# Standardize data
# Apply PCA
# Analyze variance captured by components
# Visualize reduced data in 2D

# Main goal:
# Reduce dimensions while preserving maximum information
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)

# Convert categorical data to numerical (One-Hot Encoding)
cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
df_encoded = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols) # Converts categories into binary columns.
df_encoded = df_encoded.astype(int)

# Separate features (X) and target (y)
# We drop 'target' because PCA is unsupervised
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data Standardized. Mean is now approx 0 and Variance is 1.")


# Initialize PCA - we calculate all components to see the full variance spread
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"Original number of features: {X.shape[1]}")
print(f"PCA reduced representation shape: {X_pca.shape}")

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='g', label='Individual Variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Variance')

plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Threshold')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1', alpha=0.7)

plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('PCA: 2D Projection of Heart Disease Dataset')
plt.legend(title='Condition', labels=['Healthy', 'Heart Disease'])
plt.grid(True)
plt.show()

# PCA Lab Explanation Script

# “Today I performed dimensionality reduction using PCA on the Heart Disease dataset.

# First, I imported required libraries like Pandas, NumPy, Matplotlib, Seaborn, StandardScaler from sklearn.preprocessing, and PCA from sklearn.decomposition.

# ```python
# import pandas as pd
# import numpy as np
# ```

# Pandas is used for dataframe handling and NumPy is used for numerical operations.

# ```python
# import matplotlib.pyplot as plt
# import seaborn as sns
# ```

# These libraries are used for visualization.

# ```python
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# ```

# StandardScaler is used for feature scaling and PCA is used for dimensionality reduction.

# ---

# Next, I loaded the dataset using:

# ```python
# df = pd.read_csv(url)
# ```

# `read_csv()` reads the CSV file and converts it into a dataframe.

# ---

# Then I performed one-hot encoding on categorical columns.

# ```python
# cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
# df_encoded = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)
# ```

# I used `get_dummies()` because PCA works only on numerical data.

# Categorical columns like chest pain type and thalassemia contain categories, and one-hot encoding converts them into binary columns.

# For example:

# `cp`

# becomes:

# `cp_0, cp_1, cp_2, cp_3`

# This avoids false ordering relationships between categories.

# ---

# Then I converted boolean values into integers.

# ```python
# df_encoded = df_encoded.astype(int)
# ```

# This converts True/False into 1/0.

# ---

# Next, I separated features and target.

# ```python
# X = df_encoded.drop('target', axis=1)
# y = df_encoded['target']
# ```

# `X` contains input features and `y` contains labels.

# I dropped the target column because PCA is an unsupervised learning algorithm and does not use class labels.

# `axis=1` means column operation.

# ---

# After that, I standardized the data.

# ```python
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# ```

# Standardization is very important before PCA because PCA is sensitive to feature scale.

# Features with larger values can dominate the variance calculation.

# StandardScaler transforms data using:

# genui{"math_block_widget_always_prefetch_v2":{"content":"z=\frac{x-\mu}{\sigma}"}}

# where:

# * x is original value
# * μ is mean
# * σ is standard deviation

# After scaling, mean becomes approximately 0 and variance becomes 1.

# `fit_transform()` first learns the mean and standard deviation using `fit()`, then applies scaling using `transform()`.

# ---

# Then I created the PCA object.

# ```python
# pca = PCA()
# ```

# This initializes PCA.

# Since I did not specify `n_components`, PCA keeps all principal components.

# ---

# Next, I applied PCA transformation.

# ```python
# X_pca = pca.fit_transform(X_scaled)
# ```

# PCA internally:

# * computes covariance matrix
# * finds eigenvectors and eigenvalues
# * identifies directions of maximum variance

# These new directions are called principal components.

# `fit_transform()` learns the components and projects data onto the new feature space.

# ---

# Then I checked the number of features.

# ```python
# X.shape[1]
# ```

# `shape[1]` gives the number of columns or features.

# ---

# Next, I calculated explained variance ratio.

# ```python
# explained_variance = pca.explained_variance_ratio_
# ```

# This tells how much variance each principal component captures.

# For example:

# * PC1 may capture 40%
# * PC2 may capture 20%

# ---

# Then I calculated cumulative variance.

# ```python
# cumulative_variance = np.cumsum(explained_variance)
# ```

# `cumsum()` calculates running total of variance.

# This helps determine how many principal components are needed to preserve most of the information.

# ---

# After that, I plotted the scree plot.

# ```python
# plt.bar()
# plt.step()
# ```

# The bar graph shows individual explained variance and the step graph shows cumulative variance.

# I also plotted a 90% threshold line.

# ```python
# plt.axhline(y=0.9)
# ```

# This helps identify the number of principal components required to retain 90% variance.

# ---

# Finally, I visualized the dataset in 2D PCA space.

# ```python
# sns.scatterplot(
# x=X_pca[:,0],
# y=X_pca[:,1],
# hue=y
# )
# ```

# `X_pca[:,0]` represents Principal Component 1 and `X_pca[:,1]` represents Principal Component 2.

# `hue=y` colors points according to target class.

# This visualization helps observe class separation in reduced dimensions.

# ---

# Conclusion:

# Using PCA, I transformed the dataset into principal components that preserve maximum variance while reducing redundancy between features. PCA helps in dimensionality reduction, visualization, reducing computation, and improving model efficiency.”
