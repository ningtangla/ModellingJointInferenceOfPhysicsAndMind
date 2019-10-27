import numpy as np

def calculateIncludedAngle(vector1, vector2):
    includedAngle = np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1]))
    return includedAngle

def compose2DCoordinateTranslateMatrix(translateVector):

    translateMat = np.mat([[1, 0, 0], [0, 1, 0], [-translateVector[0], -translateVector[1], 1]])
    return translateMat

def compose2DCoordinateRotateMatrix(rotateAngle):
    rotateMat = np.mat([[np.cos(rotateAngle), np.sin(rotateAngle), 0], [-np.sin(rotateAngle), np.cos(rotateAngle), 0], [0, 0, 1]])
    return rotateMat

def compose2DCoordinateScaleMatrix(scale):
    scaleMat = np.mat([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    return scaleMat

def transposeCoordinate(coordinate, transposeMat):
    coordinateMat = np.mat([coordinate[0], coordinate[1], 1])
    transposedCoordinate = np.array(coordinateMat * transposeMat)[0][:2]
    return transposedCoordinate

class OffsetMasterStates:
    def __init__(self, wolfId, masterId, ropePartIndexes, positionIndex, stateIndex, masterDelayStep):
        self.wolfId = wolfId
        self.masterId = masterId
        self.ropePartIndexes = ropePartIndexes
        self.positionIndex = positionIndex
        self.stateIndex = stateIndex
        self.masterDelayStep = masterDelayStep

    def __call__(self, traj):
        traj = np.array(traj)
        originalMasterStates = np.array([np.array(timeStep)[self.stateIndex][self.masterId] for timeStep in traj[self.masterDelayStep:]])
        originalWolfStates = np.array([np.array(timeStep)[self.stateIndex][self.wolfId] for timeStep in traj[self.masterDelayStep:]])
        originalRopePartsStates = np.array([np.array(timeStep)[self.stateIndex][self.ropePartIndexes] for timeStep in traj[self.masterDelayStep:]])
        masterStates = np.array([np.array(timeStep)[self.stateIndex][self.masterId] for timeStep in traj[: -self.masterDelayStep]])

        originalWolfPoses = originalWolfStates[:,self.positionIndex]
        masterWolfVectors = masterStates[:,self.positionIndex] - originalWolfStates[:,self.positionIndex]
        originalMasterWolfVectors = originalMasterStates[:,self.positionIndex] - originalWolfStates[:,self.positionIndex]

        translateMats = [compose2DCoordinateTranslateMatrix(wolfPos) for wolfPos in originalWolfPoses]
        rotateAngles = [calculateIncludedAngle(masterWolfVector, originalMasterWolfVector) for masterWolfVector, originalMasterWolfVector in zip(masterWolfVectors, originalMasterWolfVectors)]
        rotateMats = [compose2DCoordinateRotateMatrix(rotateAngle) for rotateAngle in rotateAngles]
        scales = [np.linalg.norm(masterWolfVector)/np.linalg.norm(originalMasterWolfVector) for masterWolfVector, originalMasterWolfVector in zip(masterWolfVectors, originalMasterWolfVectors)]

        scaleMats = [compose2DCoordinateScaleMatrix(min(1.3, scale)) for scale in scales]
        scaleBackMats = [compose2DCoordinateScaleMatrix(1/scale) for scale in scales]
        translateBackMats = [compose2DCoordinateTranslateMatrix(-wolfPos) for wolfPos, scale in zip(originalWolfPoses,scales)]
        print(np.max(scales))
        transposeMats = [translateMat*scaleMat*rotateMat*translateBackMat for translateMat, rotateMat, scaleMat, scaleBackMat, translateBackMat in zip(translateMats, rotateMats, scaleMats, scaleBackMats,translateBackMats)]
        transposedRopePartPoses = np.array([[transposeCoordinate(ropePartState[self.positionIndex], transposeMat) for ropePartState in ropePartsState] for ropePartsState, transposeMat in zip(originalRopePartsStates, transposeMats)])

        allAgentsStates = np.array([timeStep[self.stateIndex] for timeStep in traj[self.masterDelayStep:]])
        allAgentsStates[:, self.masterId] = masterStates
        allAgentsStates[:, min(self.ropePartIndexes):max(self.ropePartIndexes) + 1, min(self.positionIndex):max(self.positionIndex) + 1] = transposedRopePartPoses
        return allAgentsStates


class FindCirlceBetweenWolfAndMaster:
    def __init__(self, wolfId, masterId, stateIndex, positionIndex, timeWindow, angleDeviation):
        self.wolfId = wolfId
        self.masterId = masterId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex
        self.timeWindow = timeWindow
        self.angleDeviation = angleDeviation

    def __call__(self, traj):
        traj = np.array(traj)

        wolfVectorlist = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]

        ropeVectorlist = [traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.masterId][self.positionIndex] for i in range(1, len(traj))]

        ropeWolfAngleList = np.array([abs(calculateIncludedAngle(v1, v2)) for (v1, v2) in zip(wolfVectorlist, ropeVectorlist)])
        filterlist = [findCircleMove(ropeWolfAngleList[i:i + self.timeWindow], self.angleDeviation) for i in range(0, len(ropeWolfAngleList) - self.timeWindow + 1)]
        isValid = np.all(filterlist)
        return isValid


def findCircleMove(anglelist, angleVariance):
    lower = np.pi / 2 - angleVariance
    upper = np.pi / 2 + angleVariance
    filterAnglelist = list(filter(lambda x: x <= upper and x >= lower, anglelist))
    return len(filterAnglelist) < len(anglelist)

class CountCollisionForOffsetMaster:
    def __init__(self,masterId,wolfId,stateIndex, positionIndex,collisionRadius,isCollision):
        self.masterId =masterId
        self.wolfId=wolfId 
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex
        self.isCollision=isCollision
        self.collisionRadius=collisionRadius
    def __call__(self,traj):
        masterCoordinateList = [traj[i][self.masterId][self.positionIndex] for i in range(len(traj))]
        wolfCoordinateList = [traj[i][self.wolfId][self.positionIndex] for i in range(len(traj))]
        collisionList= [self.isCollision(coor1,coor2,self.collisionRadius) for (coor1, coor2) in zip(sheepCoordinateList, wolfCoordinateList)]

        collisionNumber = len(np.where(collisionList)[0])
        collisionRatio = collisionNumber / len(traj)
        return collisionRatio

class CountSheepCrossRope:
    def __init__(self,sheepId, wolfId, masterId,stateIndex, positionIndex,tranformCoordinates,isCrossAxis):
        self.sheepId =sheepId
        self.wolfId = wolfId
        self.masterId=masterId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex
        self.tranformCoordinates=tranformCoordinates
        self.isCrossAxis=isCrossAxis
    def __call__(self, traj):
        ropeVectorList = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] - traj[i][self.stateIndex][self.masterId][self.positionIndex] for i in range(len(traj)-1)]
        sheepMasterVectorList = [traj[i][self.stateIndex][self.sheepId][self.positionIndex] - traj[i][self.stateIndex][self.masterId][self.positionIndex] for i in range(len(traj)-1)]
        tanformeredSheepCoordatesList =[self.tranformCoordinates(xaxis, sheepvector) for (xaxis, sheepvector) in zip(ropeVectorList, sheepMasterVectorList)]
        crossList=[self.isCrossAxis(tanformeredSheepCoordatesList[i],tanformeredSheepCoordatesList[i+1]) for i in range(len(tanformeredSheepCoordatesList)-1)]
        crossNumber=len(np.where(crossList)[0])
        crossRatio = crossNumber/ len(traj)
        return crossRatio