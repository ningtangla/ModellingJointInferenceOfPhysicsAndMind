import numpy as np


def createSymmetricVector(symmetricAxis, originalVector):
    orthogonalVector = symmetricAxis.dot(originalVector) * symmetricAxis / np.power(np.power(np.power(symmetricAxis, 2).sum(),0.5), 2) - originalVector
    symmetricVector = originalVector + 2*orthogonalVector
    return symmetricVector
