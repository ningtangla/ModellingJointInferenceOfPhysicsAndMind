def createSymmetricVector(symmetricAxis, originalVector):
    orthogonalVector = symmetricAxis.dot(originalVector) * symmetricAxis / np.power(computeVectorNorm(symmetricAxis),2) - originalVector
    symmetricVector = originalVector + 2*orthogonalVector
return symmetricVector
