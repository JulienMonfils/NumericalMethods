import numpy as np
from numpy import linalg as lng

def QRMethod(matrix, precision, maxIteration):
    """
    matrix      : A square matrix (numpy array)
    precision   : Precision used to compute the eigenvectors
    maxIteration: The maximum number of QR iteration done by the algorithm

    return  : The eigenvalues and eigenvectors of the given matrix
    """

    iteration = 0
    delta = np.inf

    Ai = matrix         #The new matrix computed at each iteration
    while (iteration <= maxIteration and delta > precision):
        Q, R = lng.qr(Ai)
        Ai = np.dot(R, Q)

        iteration += 1
        print(Ai)
    return np.diagonal(Ai)

A = np.array([[1,2],[2,1]])

print(QRMethod(A, 5, 15))
