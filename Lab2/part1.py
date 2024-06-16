import numpy as np


def values_vectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    check = []

    for i in range(len(eigenvalues)):
        lambda_v = eigenvalues[i] * eigenvectors[:, i]
        A_v = np.dot(matrix, eigenvectors[:, i])
        check.append(np.allclose(A_v, lambda_v))
    return eigenvalues, eigenvectors, check


matrix = np.array([[4, -2],
                   [1, 1]])

eigenvalues, eigenvectors, check = values_vectors(matrix)
print("Own values:", eigenvalues)
print("Own vectors:\n", eigenvectors)

for i, check in enumerate(check):
    print(f"Check {eigenvalues[i]}: {check}")