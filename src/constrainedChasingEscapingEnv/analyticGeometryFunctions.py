import numpy as np


def transiteCartesianToPolar(vector):
    return np.arctan2(vector[1], vector[0])


def transitePolarToCartesian(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def computeVectorNorm(vector):
    return np.linalg.norm(vector)


def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(vector1, vector2.T)
    innerProduct = vectoriseInnerProduct
    unclipRatio = innerProduct/(computeVectorNorm(vector1) * computeVectorNorm(vector2))
    ratio = np.clip(unclipRatio, -1.0, 1.0)# float precision probblem as enmin report
    angle = np.arccos(ratio)
    return angle

def computeVectorNorm(vector):
    L2Norm = np.linalg.norm(vector, ord = 2)
    return L2Norm

