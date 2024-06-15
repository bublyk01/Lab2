import numpy as np


def values_vectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors


matrix = np.array([[4, -2],
                   [1, 1]])

eigenvalues, eigenvectors = values_vectors(matrix)
print("Own values:", eigenvalues)
print("Own vectors:\n", eigenvectors)
