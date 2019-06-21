import numpy as np 

def calculateAngle(firstVector, secondVector):
    firstVectorArray = np.array(firstVector)
    secondVectorArray = np.array(secondVector)

    length_firstVector = np.sqrt(firstVectorArray.dot(firstVectorArray))
    length_secondVector = np.sqrt(secondVectorArray.dot(secondVectorArray))

    if length_firstVector * length_secondVector == 0:
        return -np.inf

    cosAngle = firstVectorArray.dot(secondVectorArray)/ (length_firstVector * length_secondVector)
    angleInBetween = np.arccos(cosAngle)
    return angleInBetween