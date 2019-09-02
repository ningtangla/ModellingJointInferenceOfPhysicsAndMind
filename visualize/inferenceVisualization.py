import numpy as np
import pygame as pg
import pandas as pd
import os
import itertools as it
from pylab import plt

class SaveImage:
    def __init__(self, imageFolderName):
        self.imageFolderName = imageFolderName

    def __call__(self, screen):
        currentDir = os.getcwd()
        parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
        saveImageDir = os.path.join(os.path.join(parentDir, 'demo'), self.imageFolderName)
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        fileCount = len([name for name in os.listdir(saveImageDir) if os.path.isfile(os.path.join(saveImageDir, name))])
        pg.image.save(screen, saveImageDir + '/' + format(fileCount, '05') + ".png")

class GetChasingRoleColor:
    def __init__(self, roleColor, roleIndex):
        self.roleColor = roleColor
        self.roleIndex = roleIndex

    def __call__(self, agentIndex, resultGroupedByChasingIndices):
        indexList = resultGroupedByChasingIndices['mind']
        chasingFilter = [chasingCondition[agentIndex] == self.roleIndex for chasingCondition in indexList]
        chasingIndexList = [i for i, index in zip(range(len(indexList)), chasingFilter) if index]
        agentPosterior = sum(
            [resultGroupedByChasingIndices.iloc[chasingIndex]['posterior'] for chasingIndex in
             chasingIndexList])
        newColor = tuple(agentPosterior * np.array(self.roleColor))
        return newColor


class GetChasingResultColor:
    def __init__(self, getRolesColor):
        self.getRolesColor = getRolesColor

    def __call__(self, agentIndex, resultGroupedByChasingIndices):
        rolesColor = [getColor(agentIndex, resultGroupedByChasingIndices) for getColor in self.getRolesColor]
        resultColor = tuple(sum(np.array(rolesColor)))
        return resultColor


class ColorChasingPoints:
    def __init__(self, getChasingResultColor):
        self.getChasingResultColorColor = getChasingResultColor

    def __call__(self, resultGroupedByChasingIndices):
        agentCount = len(resultGroupedByChasingIndices['mind'][0])
        colorList = [self.getChasingResultColorColor(agentIndex, resultGroupedByChasingIndices) for agentIndex in
                     range(agentCount)]
        return colorList


class AdjustPullingLineWidth:
    def __init__(self, minWidth = 0, maxWidth = 5):
        self.minWidth = minWidth
        self.maxWidth = maxWidth

    def __call__(self, resultGroupedByPullingIndices):
        pullingIndexList = resultGroupedByPullingIndices['physics']
        lineWidthDict = dict()
        for index in range(len(pullingIndexList)):
            pullingIndex = pullingIndexList[index]
            agentPosterior = resultGroupedByPullingIndices['posterior'].values[index]

            pullingWidth = int(np.round(self.minWidth + agentPosterior * (self.maxWidth - self.minWidth)))
            pulledAgentID, pullingAgentID = np.where(np.array(pullingIndex) == 'pulled')[0]

            lineWidthDict[(pulledAgentID, pullingAgentID)] = pullingWidth

        return lineWidthDict


class DrawInferenceResultWithPull:
    def __init__(self, inferenceIndex, gridPixelSize, drawGrid, drawCirclesAndLines,
                 colorChasingPoints, adjustPullingLineWidth):
        self.inferenceIndex = inferenceIndex

        self.gridPixelSize = gridPixelSize
        self.drawGrid = drawGrid
        self.drawCirclesAndLines = drawCirclesAndLines

        self.colorChasingPoints = colorChasingPoints
        self.adjustPullingLineWidth = adjustPullingLineWidth

    def __call__(self, state, posterior):
        screen = self.drawGrid()
        pointsCoord = state
        pointsLocation = [list(np.array(pointCoord) * self.gridPixelSize - self.gridPixelSize // 2)
                          for pointCoord in pointsCoord]

        inferenceDf = pd.DataFrame(index= self.inferenceIndex)
        inferenceDf['posterior'] = posterior

        resultGroupedByChasingIndices = inferenceDf.groupby('mind').sum().reset_index()
        pointsColor = self.colorChasingPoints(resultGroupedByChasingIndices)

        resultGroupedByPullingIndices = inferenceDf.groupby('physics').sum().reset_index()
        linesWidth = self.adjustPullingLineWidth(resultGroupedByPullingIndices)

        screen = self.drawCirclesAndLines(screen, pointsLocation, pointsColor, linesWidth)

        return screen

class DrawContinuousInferenceResultNoPull:
    def __init__(self, numOfAgents, inferenceIndex, drawState, scaleState,
                 colorChasingPoints, adjustFPS, saveImage):
        self.numOfAgents = numOfAgents
        self.inferenceIndex = inferenceIndex
        self.drawState = drawState
        self.scaleState = scaleState
        self.colorChasingPoints = colorChasingPoints
        self.adjustFPS = adjustFPS
        self.saveImage = saveImage

    def __call__(self, currentState, nextState, posterior):
        currentPosition = self.scaleState(currentState)
        nextPosition =  self.scaleState(nextState)
        print("currentPosition: ", currentPosition)

        inferenceDf = pd.DataFrame(index=self.inferenceIndex)
        inferenceDf['posterior'] = posterior

        resultGroupedByChasingIndices = inferenceDf.groupby('mind').sum().reset_index()
        circleColorList = self.colorChasingPoints(resultGroupedByChasingIndices)
	
        positionsList = self.adjustFPS(currentPosition, nextPosition)

        for positionIndex in range(len(positionsList)):
            screen = self.drawState(self.numOfAgents, positionsList[positionIndex], circleColorList)
            if self.saveImage is not None:
                self.saveImage(screen)

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

class TransposeRopePosesInState:
    def __init__(self, wolfId, masterId, ropePartIndexes, positionIndex):
        self.wolfId = wolfId
        self.masterId = masterId
        self.ropePartIndexes = ropePartIndexes
        self.positionIndex = positionIndex

    def __call__(self, state, tiedAgentIds):
        notBasePointAgentId, basePointAgentId = tiedAgentIds
        stateArray = np.array(state)

        originalMasterStates = np.array([stateArray[self.masterId]])
        originalWolfStates = np.array([stateArray[self.wolfId]])
        originalRopePartsStates = np.array([stateArray[self.ropePartIndexes]])
        originalBasePointAgentStates = np.array([stateArray[basePointAgentId]])
        originalNotBasePointAgentStates = np.array([stateArray[notBasePointAgentId]])

        originalBasePointAgentPoses = originalBasePointAgentStates[:,self.positionIndex]
        newVectors = originalNotBasePointAgentStates[:,self.positionIndex] - originalBasePointAgentStates[:,self.positionIndex]
        if basePointAgentId == self.wolfId:
            originalVectors = originalMasterStates[:,self.positionIndex] - originalWolfStates[:,self.positionIndex]
        if basePointAgentId == self.masterId:
            originalVectors = originalWolfStates[:,self.positionIndex] - originalMasterStates[:,self.positionIndex]

        translateMats = [compose2DCoordinateTranslateMatrix(basePointAgentPos) for basePointAgentPos in originalBasePointAgentPoses]
        rotateAngles = [calculateIncludedAngle(newVector, originalVector) for newVector, originalVector in zip(newVectors, originalVectors)]
        rotateMats = [compose2DCoordinateRotateMatrix(rotateAngle) for rotateAngle in rotateAngles]
        scales = [np.linalg.norm(newVector)/np.linalg.norm(originalVector) for newVector, originalVector in zip(newVectors, originalVectors)]

        scaleMats = [compose2DCoordinateScaleMatrix(min(1.6, scale)) for scale in scales]
        scaleBackMats = [compose2DCoordinateScaleMatrix(1/scale) for scale in scales]
        translateBackMats =  [compose2DCoordinateTranslateMatrix(-basePointAgentPos) for basePointAgentPos in originalBasePointAgentPoses]
        transposeMats = [translateMat*scaleMat*rotateMat*translateBackMat for translateMat, rotateMat, scaleMat, scaleBackMat, translateBackMat in zip(translateMats, rotateMats, scaleMats, scaleBackMats,translateBackMats)]
        transposedRopePartPoses = np.array([[transposeCoordinate(ropePartState[self.positionIndex], transposeMat) for ropePartState in ropePartsState] for ropePartsState, transposeMat in zip(originalRopePartsStates, transposeMats)])
        
        stateArray[min(self.ropePartIndexes):max(self.ropePartIndexes) + 1] = transposedRopePartPoses[0]
        return stateArray


class DrawContinuousInferenceResultWithPull:
    def __init__(self, numOfAgents, tiedMinds, inferenceIndex, drawStateWithRopeInProbability, transposeRopePos, scaleState, 
                 colorChasingPoints, adjustFPS, saveImage):
        self.numOfAgents = numOfAgents
        self.tiedMinds = tiedMinds
        self.numOfTiedAgents = len(tiedMinds)
        self.inferenceIndex = inferenceIndex
        self.drawStateWithRopeInProbability = drawStateWithRopeInProbability
        self.transposeRopePos = transposeRopePos
        self.scaleState = scaleState
        self.colorChasingPoints = colorChasingPoints
        self.adjustFPS = adjustFPS
        self.saveImage = saveImage

    def __call__(self, currentState, nextState, posterior):
        inferenceDf = pd.DataFrame(index=self.inferenceIndex)
        inferenceDf['posterior'] = posterior
        resultGroupedByChasingIndices = inferenceDf.groupby('mind').sum().reset_index()
        mindValues = [np.array(mind) for mind in resultGroupedByChasingIndices['mind'].values]
        
        possibleTiedAgents = list(it.combinations(range(self.numOfAgents), self.numOfTiedAgents))
        leagelIndexTiedAgents = np.nonzero([np.all(np.argsort(possibleTiedAgent) == np.array(range(self.numOfTiedAgents))) for possibleTiedAgent in possibleTiedAgents])[0]
        tiedAgentIdPairs = np.array(possibleTiedAgents)[leagelIndexTiedAgents]
        currentPosition = self.scaleState(currentState)
        nextPosition =  self.scaleState(nextState)
        print("currentPosition: ", currentPosition)

        circleColorList = self.colorChasingPoints(resultGroupedByChasingIndices)

        positionsList = self.adjustFPS(currentPosition, nextPosition)

        for positionIndex in range(len(positionsList)):
            transposedStates = [self.transposeRopePos(positionsList[positionIndex], tiedAgentId) for tiedAgentId in tiedAgentIdPairs] 
            tiedMindLeagelIndexs = [[np.all(np.isin(mindValue[list(tiedAgentId)], np.array(self.tiedMinds))) for mindValue in mindValues] for tiedAgentId in tiedAgentIdPairs] 
            ropeWidthRatios = [np.sum(resultGroupedByChasingIndices['posterior'][np.nonzero(tiedMindLeagel)[0]]) for tiedMindLeagel in tiedMindLeagelIndexs]
            screen = self.drawStateWithRopeInProbability(transposedStates, tiedAgentIdPairs, circleColorList, ropeWidthRatios)
            if self.saveImage is not None:
                self.saveImage(screen)

class PlotInferenceProb:
    def __init__(self, xVariableName, yVaraibleName, groupIndex):
        self.xVariableName = xVariableName
        self.yVaraibleName = yVaraibleName
        self.groupIndex = groupIndex

    def __call__(self, inferenceDf, graphIndex, plotname):
        # print(inferenceDf)
        resultDf = inferenceDf.groupby(self.groupIndex).sum()
        print(resultDf)
        graph = resultDf.T.plot()
        graph.set_xlabel(self.xVariableName)
        graph.set_ylabel(self.yVaraibleName)
        plt.ylim([0, 1])
        plt.title(self.groupIndex+ ": " + plotname + '_dataIndex'+ str(graphIndex))
        dirName = os.path.dirname(__file__)
        plotPath = os.path.join(dirName, '..', 'demo')
        plt.savefig(os.path.join(plotPath, self.groupIndex + plotname + 'data'+ str(graphIndex)))
        plt.show()
