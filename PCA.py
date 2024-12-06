import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets as ds

# Load the iris dataset
data = ds.load_iris(as_frame=True).frame

# Define the flower species
species_names = ["I. setosa", "I. versicolor", "I. virginica"]

# Select the features
# In X we ignore the target variable (flower species)
X = data.iloc[:, 0:-1].values
# In y we store the target variable
y = data.iloc[:, -1].values

# Standardize the data (mean=0, standard deviation=1)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_standardized = (X - X_mean) / X_std

# Build the covariance matrix
# rowvar=False -> the columns are the variables
S = np.cov(X_standardized, rowvar=False)

# Obtain the eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(S)

# Sort the eigenvectors by decreasing eigenvalues 
sorted_idxs = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idxs]
eigenvectors = eigenvectors[:, sorted_idxs]

# Project the data onto the new space
Y = X_standardized @ eigenvectors

# Calculate the explained variance
explained_variance = eigenvalues / np.sum(eigenvalues)

# Print the explained variance
print(f"Explained variance by each principal component:")
for i, ratio in enumerate(explained_variance):
    print(f"PC{i + 1}: {ratio * 100:.2f}%")

# Plot the PCA
for label, species in zip(np.unique(y), species_names):
    plt.scatter(Y[y == label, 0],
                Y[y == label, 1],
                label=rf"$\it{{{species}}}$")

plt.xlabel(f"PC1 ({explained_variance[0] * 100:.2f}% of explained variance)")
plt.ylabel(f"PC2 ({explained_variance[1] * 100:.2f}% of explained variance)")
plt.title("PCA of the Iris Dataset")
plt.legend()
plt.grid()
plt.show()
