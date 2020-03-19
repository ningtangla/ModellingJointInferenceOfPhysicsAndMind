import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product
import pygame as pg
from pygame.color import THECOLORS

from src.constrainedChasingEscapingEnv.envNoPhysics import  TransiteForNoPhysicsWithCenterControlAction, Reset,IsTerminal,StayInBoundaryByReflectVelocity,UnpackCenterControlAction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist,Expand
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy
from src.episode import  SampleAction, chooseGreedyAction,Render,chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


def main():

    dirName = os.path.dirname(__file__)
    maxRunningSteps = 100
    numSimulations = 200
    killzoneRadius = 30
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius}
    trajectoryDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsTwoWolves', 'trajectories')
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle,fuzzySearchParameterNames)

    # para = {'numSimulations':numSimulations }
    iterationIndex=0
    para = {'iterationIndex':iterationIndex }
    allTrajectories = loadTrajectories(para)



    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
    circleSize = 10
    saveImage = False
    saveImageDir = os.path.join(dirName, '..','..', '..', 'data','demoImg')
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    xBoundary = [0,600]
    yBoundary = [0,600]
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    numOfAgent=3
    posIndex = [0, 1]

    render = Render(numOfAgent, posIndex,screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)
    stateIndex=0
    drawTrajectory = DrawTrajectoryWithRender(stateIndex,render)


    for dataIndex in range(len(allTrajectories)):
        trajectory = allTrajectories[dataIndex]
        del trajectory[-1]
        drawTrajectory(trajectory)


class DrawTrajectoryWithRender:
    def __init__(self, stateIndex,render):
        self.stateIndex=stateIndex
        self.render = render


    def __call__(self, trajectory):

        for runningStep in range(len(trajectory)):
            state=trajectory[runningStep][self.stateIndex]
            self.render(state, runningStep)

        return trajectory
if __name__ == '__main__':
    main()
