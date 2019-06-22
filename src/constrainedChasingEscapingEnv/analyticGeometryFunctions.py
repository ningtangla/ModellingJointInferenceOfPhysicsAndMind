import numpy as np

def transiteCartesianToPolar(vector):
    return np.arctan2(vector[1], vector[0])

def transitePolarToCartesian(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(vector1, vector2.T)
    innerProduct = vectoriseInnerProduct
    angle = np.arccos(innerProduct/(computeVectorNorm(vector1) * computeVectorNorm(vector2)))
    return angle

def computeVectorNorm(vector):
    L2Norm = np.linalg.norm(vector, ord = 2)
    return L2Norm
