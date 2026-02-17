import numpy as np
def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """

    A = np.asarray(matrix)
    
    if matrix.shape[0] != matrix.shape[1]:
        return None
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues

