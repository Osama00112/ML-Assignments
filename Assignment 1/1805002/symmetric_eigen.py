import numpy as np
import matplotlib.pyplot as plt

def random_invertible_symmetric_matrix(n):
    random_matrix = np.random.rand(n, n)
    random_matrix = (random_matrix * 100).astype(int)
    
    # source: https://stackoverflow.com/questions/10806790/generating-symmetric-matrices-in-numpy
    A = random_matrix + random_matrix.T
    print("\nSymmetric matrix A: \n", A)
    
    # creating diagonal dominant matrix
    # source: https://stackoverflow.com/questions/73426718/generating-invertible-matrices-in-numpy-tensorflow
    # proof of invertibility: https://en.m.wikipedia.org/wiki/Diagonally_dominant_matrix?fbclid=IwAR3p_u09EV8XQliSz5DRTOxMH1rE9fKB0qqsMdzrDYx0LMrmsn8Knw-HIIw
    
    row_sum_matrix = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, row_sum_matrix)
    print("\nSymmetric + Invertible matrix A: \n", A)
    
    return A

def check_decomposition(n):
    A = random_invertible_symmetric_matrix(n)
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
    

check_decomposition(7)    
    
        
    