##############################################################################
# PCA.py
# 
# Análisis de Componentes Principales (PCA)
# Programación y Técnicas Computacionales Avanzadas
# 
# Este programa realiza un análisis de componentes principales (PCA) sobre el
# conjunto de datos de la flor Iris. Por consola se muestra el porcentaje de
# varianza explicada por cada componente principal. Además, se generan dos
# gráficas: una en 2D y otra en 3D, mostrando la representación gráfica de las
# las dos primeras componentes principales y las tres primeras componentes
# principales, respectivamente.
#
# Autor: Francisco Javier Cordero Felipe
# Fecha: 20/12/2024
# Versión: 1.2
##############################################################################

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
    print(f"* PC{i + 1}: {ratio * 100:.2f}%")

# Plot the PCA
fig = plt.figure(figsize=(12, 6))

# 2D plot
ax1 = fig.add_subplot(121)
for label, species in zip(np.unique(y), species_names):
    ax1.scatter(Y[y == label, 0],
                Y[y == label, 1],
                label=rf"$\it{{{species}}}$")

ax1.set_xlabel(f"PC1 ({explained_variance[0] * 100:.2f}%)")
ax1.set_ylabel(f"PC2 ({explained_variance[1] * 100:.2f}%)")
ax1.set_title("2D PCA - Iris Dataset")
ax1.legend()
ax1.grid()

# 3D plot
ax2 = fig.add_subplot(122, projection='3d')
for label, species in zip(np.unique(y), species_names):
    ax2.scatter(Y[y == label, 0],
               Y[y == label, 1],
               Y[y == label, 2],
               label=rf"$\it{{{species}}}$")
    
ax2.set_xlabel(f"PC1 ({explained_variance[0] * 100:.2f}%)")
ax2.set_ylabel(f"PC2 ({explained_variance[1] * 100:.2f}%)")
ax2.set_zlabel(f"PC3 ({explained_variance[2] * 100:.2f}%)")
ax2.set_title("3D PCA - Iris Dataset")
ax2.legend()

plt.show()
