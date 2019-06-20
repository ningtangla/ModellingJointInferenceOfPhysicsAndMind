import numpy as np

def transiteCartesianToPolar(vector):
    return np.arctan2(vector[1], vector[0])

def transitePolarToCartesian(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(vector1, vector2.T)
    if np.ndim(vectoriseInnerProduct) > 0:
        innerProduct = vectoriseInnerProduct.diagonal()
    else:
        innerProduct = vectoriseInnerProduct
    angle = np.arccos(innerProduct/(computeVectorNorm(vector1) * computeVectorNorm(vector2)))
    return angle

def computeVectorNorm(vector):
    return np.power(np.power(vector, 2).sum(axis = 0), 0.5)

def getSymmetricVector(symmetricAxis, originalVector):
    orthogonalVector = symmetricAxis.dot(originalVector) * symmetricAxis / np.power(computeVectorNorm(symmetricAxis),2) - originalVector
    symmetricVector = originalVector + 2*orthogonalVector
    return symmetricVector

def calculateCrossEntropy(prediction, target, episilon = 1e-12):
    ce = -1 * sum([target[index] * np.log(prediction[index]+episilon) for index in range(len(prediction))])
    return ce