import numpy as np
from matplotlib import pyplot as plt
import random as rd
import numpy.linalg as lng

def powerIteration(A, nIteration, initialGuess = None):
    """
    Return la valeur propre dominante et son vecteur propre unitaire associé

    A               : Matrice carrée
    nIteration      : Nombre d'itération de la méthode
    initialGuess    : Là où commence la méthode (un vector random si non fourni)
    """


    #Si pas d'approximation, on mets un vecteur random
    if initialGuess == None:
        initialGuess = np.random.rand(len(A))

    eigenVector = initialGuess

    #itérations de la méthode
    for k in range(nIteration):
        Ax = np.dot(A, eigenVector)
        eigenVector = Ax / lng.norm(Ax)

    eigenValue = lng.norm(np.dot(A, eigenVector))
    return (eigenValue, eigenVector)

A = np.array([[1,2,4],[3,5,4],[8,6,3]])


print(powerIteration(A, 150))
