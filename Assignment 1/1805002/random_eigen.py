import numpy as np
import matplotlib.pyplot as plt

def random_invertible_matrix(n):
    A = np.random.rand(n, n)
    A = (A * 100).astype(int)
    while np.linalg.det(A) == 0:
        A = np.random.rand(n, n)
        A = (A * 100).astype(int)
    print("\nRandom invertible matrix A: \n", A)
    return A


def check_decomposition(n):
    A = random_invertible_matrix(n)
    eigen_values, eigen_vectors = np.linalg.eig(A)
    print("\nEigen values: \n", eigen_values, "\n\nEigen vectors:\n", eigen_vectors)

    
    
    diagonal = np.diag(eigen_values)
    print("\nDiagonal matrix: \n", diagonal)

    V = eigen_vectors
    inv_V = np.linalg.inv(V)
    reconstructed_A = V.dot(diagonal).dot(inv_V)
    reconstructed_A = np.round(reconstructed_A)
    reconstructed_A = reconstructed_A.astype(int)
    print("\nReconstructed A: \n", reconstructed_A)

    #Checking if equal
    print("\nEquality verdict: ", np.allclose(A, reconstructed_A))
    

check_decomposition(3)










