import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import json
import pandas as pd
from matplotlib import pyplot as plt
from collections import OrderedDict
from subprocess import Popen, PIPE
import time
import pickle
from mujoco_py import load_model_from_path, MjSim
import numpy as np

from exec.evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics
from src.constrainedChasingEscapingEnv.wrappers import GetStateFromTrajectory, GetAgentPosFromTrajectory, \
    GetAgentPosFromState
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.measure import ComputeOptimalNextPos, DistanceBetweenActualAndOptimalNextPosition
from src.constrainedChasingEscapingEnv.envMujoco import TransitionFunction, IsTerminal
from src.play import agentDistToGreedyAction
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


def drawPerformanceLine(dataDf, axForDraw, steps):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        grp.plot(ax=axForDraw, label=key, y='mean', title='TrainSteps: {}'.format(steps))
        axForDraw.set_ylim([0, 0.4])


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()

    return dataSet


class GenerateTrajectoryParallel:
    def __init__(self, codeFileName, numSamples):
        self.codeFileName = codeFileName
        self.numSamples = numSamples

    def __call__(self, oneConditionDf):
        sampleIdStrings = list(map(str, range(self.numSamples)))
        indexLevelNames = oneConditionDf.index.names
        trialConditionParameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0]
                                    for levelName in indexLevelNames}

        convertNumpyToInt = lambda value: int(value) if isinstance(value, np.int64) else value
        trialConditionParameters = {key: convertNumpyToInt(value) for key, value in trialConditionParameters.items()}

        trialConditionString = json.dumps(trialConditionParameters)
        cmdList = [['python3', self.codeFileName, trialConditionString, sampleIndex] for sampleIndex in sampleIdStrings]
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]

        startTime = time.time()
        for proc in processList:
            stdOut, stdErr = proc.communicate()
            proc.wait()

        processTime = time.time() - startTime
        print('processTime for ', trialConditionParameters, processTime)

        return None


def main():
    # function to sample trajectories using parallelization
    codeFileName = 'generateTrajectoryForOneCondition.py'
    numSamples = 50
    generateTrajectoryParallel = GenerateTrajectoryParallel(codeFileName, numSamples)

    # condition variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['trainSteps'] = [0, 50, 100, 500]
    manipulatedVariables['sheepPolicyName'] = ['MCTS', 'MCTSNN']
    manipulatedVariables['numSimulations'] = [50, 200, 800]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # apply generateTrajectoryParallel for each condition
    startTime = time.time()
    toSplitFrame.groupby(levelNames).apply(generateTrajectoryParallel)
    endTime = time.time()
    print("total time for generating trajectories", (endTime - startTime))

    # wrappers for agent positions
    initTimeStep = 0
    stateIndex = 0
    getInitStateFromTrajectory = GetStateFromTrajectory(initTimeStep, stateIndex)
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    # mujoco environment
    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    dirName = os.path.dirname(__file__)
    mujocoModelPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    mujocoModel = load_model_from_path(mujocoModelPath)
    simulation = MjSim(mujocoModel)
    numSimulationFrames = 20
    transit = TransitionFunction(simulation, isTerminal, numSimulationFrames)
    getStationaryAgentAction = lambda state: agentDistToGreedyAction(stationaryAgentPolicy(state))
    sheepTransit = lambda state, action: transit(state, [action, getStationaryAgentAction(state)])

    # measurement Function
    optimalPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getSheepXPos, getWolfXPos,
                                                              computeAngleBetweenVectors)
    getOptimalAction = lambda state: agentDistToGreedyAction(optimalPolicy(state))
    computeOptimalNextPos = ComputeOptimalNextPos(getInitStateFromTrajectory, getOptimalAction, sheepTransit,
                                                  getSheepXPos)
    measurementTimeStep = 1
    getNextStateFromTrajectory = GetStateFromTrajectory(measurementTimeStep, stateIndex)
    getPosAtNextStepFromTrajectory = GetAgentPosFromTrajectory(getSheepXPos, getNextStateFromTrajectory)
    measurementFunction = DistanceBetweenActualAndOptimalNextPosition(computeOptimalNextPos,
                                                                      getPosAtNextStepFromTrajectory)

    # load trajectories and compute statistics on the trajectories
    trajectoryMaxRunningSteps = 2
    trajectoryQPosInit = (0, 0, 0, 0)
    trajectoryQPosInitNoise = 9.7
    trajectoryFixedParameters = {'maxRunningSteps': trajectoryMaxRunningSteps, 'qPosInit': trajectoryQPosInit,
                                 'qPosInitNoise': trajectoryQPosInitNoise}
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco',
                                       'parallel', 'evalTrajectories')
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadData)
    computeStatistics = ComputeStatistics(loadTrajectories, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot the statistics
    fig = plt.figure()

    numColumns = len(manipulatedVariables['trainSteps'])
    numRows = 1
    plotCounter = 1

    for key, grp in statisticsDf.groupby('trainSteps'):
        grp.index = grp.index.droplevel('trainSteps')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        drawPerformanceLine(grp, axForDraw, key)
        plotCounter += 1

    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()



