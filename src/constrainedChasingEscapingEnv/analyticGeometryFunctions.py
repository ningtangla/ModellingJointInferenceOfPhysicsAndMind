import numpy as np


def transiteCartesianToPolar(vector):
    return np.arctan2(vector[1], vector[0])


def transitePolarToCartesian(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def computeVectorNorm(vector):
    return np.linalg.norm(vector)


def computeAngleBetweenVectors(vector1, vector2):
    cosang = np.dot(vector1, vector2)
    sinang = np.linalg.norm(np.cross(vector1, vector2))
    angle = np.arctan2(sinang, cosang)
    return angle

# def computeAngleBetweenVectors(vector1, vector2):
#     vectoriseInnerProduct = np.dot(vector1, vector2.T)
#     if np.ndim(vectoriseInnerProduct) > 0:
#         innerProduct = vectoriseInnerProduct.diagonal()
#     else:
#         innerProduct = vectoriseInnerProduct
#     angle = np.arccos(innerProduct / (computeVectorNorm(vector1) * computeVectorNorm(vector2)))
#     return angle # gives inaccurate results for small angles
