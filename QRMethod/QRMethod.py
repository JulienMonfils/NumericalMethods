import numpy as np
from numpy import linalg as lng

def QRMethod(matrix, precision, maxIteration, ConvAlert = False):
    """
    matrix      : A square matrix (numpy array)
    precision   : Precision used to compute the eigenvectors
    maxIteration: The maximum number of QR iteration done by the algorithm
    ConvAlert   : if True, the method raise an error when the method does not converge in less than maxIteration iterations

    return  : The eigenvalues and eigenvectors of the given matrix
    """

    iteration = 0
    delta = np.inf

    Ai = matrix         #The new matrix computed at each iteration
    while (iteration <= maxIteration and delta > precision):
        Q, R = lng.qr(Ai)
        Ai = np.dot(R, Q)

        iteration += 1

        lowerTri = np.tril(Ai, -1)
        delta = np.sum(lowerTri)

    if iteration >= maxIteration and ConvAlert:
        raise Exception("method does not converge in " + str(maxIteration) + " iterations")
    
    eigenvalues = np.diagonal(Ai)

    return eigenvalues

A = np.array([[1,2],[2,1]])

print(QRMethod(A, 10**(-5), 15, ConvAlert=True))
