import numpy as np
from numpy import linalg as lng
from sympy import Matrix

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
    
    eigenVectors = np.zeros(np.shape(matrix))
    count = 0
    for eig in eigenValues:
        M = matrix - np.eye(len(matrix))*eig
        ns = nullSpaceSolver(M)
        print(M, ns)
        for j in ns:
            print(j[:,0], eigenVectors[:,count])
            eigenVectors[:,count] = j[:,0]
            count+=1


    

    return eigenValues, eigenVectors


def nullSpaceSolver(matrix):
    """
    matrix  : the square matrix we compute the nullSpace

    return  : the null space of the matrix
    """

    m = Matrix(matrix)
    ns = np.array(m.nullspace())

    return ns
    
   
A = np.array([[1,2],[2,1]])
B = np.array([[1,1,1],[2,2,2],[2,1,2]])
nullSpaceSolver(A)
print(QRMethod(A, 10**(-5), 15, ConvAlert=True))
