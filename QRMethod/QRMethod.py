import numpy as np
from numpy import linalg as lng
from scipy import linalg as slng

def QRMethod(matrix, precision, maxIteration, ConvAlert = False):
    """
    matrix      : A square matrix (numpy array)
    precision   : Precision used to compute the eigenvectors
    maxIteration: The maximum number of QR iteration done by the algorithm
    ConvAlert   : if True, the method raise an error when the method does not converge in less than maxIteration iterations

    return  : The eigenvalues of the given matrix
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
    eigenValues = np.diagonal(Ai)

    return eigenValues


def nullSpaceSolver(matrix):
    """
    matrix  : the square matrix we compute the nullSpace

    return  : the null space of the matrix
    """

    n = len(matrix)

    #Reduction of the matrix
    for i in range(n):
        for j in range(n-1, i, -1):
            matrix[j,:] = matrix[j,:] -  (matrix[j,i]/matrix[i,i])*matrix[i,:]
    print(matrix)
    #TODO choosing the row with the biggest values when doing the elimination (better stability)

    #TODO computing the nullSpace of a uppertriangular matrix (det(A) = 0)
    
A = np.array([[1,1],[2,2]])
B = np.array([[1,1,1],[2,2,2],[2,1,2]])
nullSpaceSolver(B)
print(QRMethod(A, 10**(-5), 15, ConvAlert=True))
