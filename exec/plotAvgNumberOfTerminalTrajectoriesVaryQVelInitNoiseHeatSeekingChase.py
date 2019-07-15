import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco

from exec.trajectoriesSaveLoad import LoadTrajectories, GetSavePath, loadFromPickle
from exec.evaluationFunctions import conditionDfFromParametersDict
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, ResetUniform, TransitionFunction
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.episode import SampleTrajectory, chooseGreedyAction


class GenerateTrajectories:
    def __init__(self, policy, numTrials, getSampleTrajectory):
        self.policy = policy
        self.numTrials = numTrials
        self.getSampleTrajectory = getSampleTrajectory

    def __call__(self, oneConditionDf):
        maxRunningSteps = oneConditionDf.index.get_level_values('maxRunningSteps')[0]
        qVelInitNoise = oneConditionDf.index.get_level_values('qVelInitNoise')[0]
        sampleTrajectory = self.getSampleTrajectory(maxRunningSteps, qVelInitNoise)
        trajectories = [sampleTrajectory(self.policy) for _ in range(self.numTrials)]

        return trajectories


class ComputeAverageNumTerminalTrajectories:
    def __init__(self, terminalTimeStep, actionIndex, generateTrajectories):
        self.terminalTimeStep = terminalTimeStep
        self.actionIndex = actionIndex
        self.generateTrajectories = generateTrajectories

    def __call__(self, oneConditionDf):
        trajectories = self.generateTrajectories(oneConditionDf)
        allTrajectoriesIsTerminal = [trajectory[self.terminalTimeStep][self.actionIndex] is None
                                     for trajectory in trajectories]
        mean = np.mean(allTrajectoriesIsTerminal)

        return mean


def main():
        manipulatedVariables = OrderedDict()
        manipulatedVariables['maxRunningSteps'] = [10, 100]
        manipulatedVariables['qVelInitNoise'] = [1, 5]
        toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)
        numTrials = 200

        # Mujoco Environment
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)

        dirName = os.path.dirname(__file__)
        physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
        killzoneRadius = 0.5
        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
        sheepActionInWolfSimulation = lambda state: (0, 0)

        # heat seeking policy
        heatSeekingPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getWolfXPos, getSheepXPos,
                                                                   computeAngleBetweenVectors)

        # policy
        policy = lambda state: [stationaryAgentPolicy(state), heatSeekingPolicy(state)]

        # sample trajectory
        qPosInit = (0, 0, 0, 0)
        qVelInit = (0, 0, 0, 0)
        qPosInitNoise = 9.7
        numAgent = 2
        getReset = lambda qVelInitNoise: ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgent, qPosInitNoise,
                                                      qVelInitNoise)
        getSampleTrajectory = lambda maxRunningSteps, qVelInitNoise: SampleTrajectory(maxRunningSteps, transit,
                                                                                      isTerminal, getReset(qVelInitNoise),
                                                                                      chooseGreedyAction)

        # function to generate trajectories
        generateTrajectories = GenerateTrajectories(policy, numTrials, getSampleTrajectory)

        # function to compute average num trials
        terminalTimeStep = -1
        actionIndex = 1
        computeAverageNumTerminalTrajectories = ComputeAverageNumTerminalTrajectories(terminalTimeStep, actionIndex,
                                                                                      generateTrajectories)

        levelNames = list(manipulatedVariables.keys())
        statisticsDf = toSplitFrame.groupby(levelNames).apply(computeAverageNumTerminalTrajectories)

        fig = plt.figure()
        axForDraw = fig.add_subplot(1, 1, 1)

        for qVelInitNoise, grp in statisticsDf.groupby('qVelInitNoise'):
            grp.index = grp.index.droplevel('qVelInitNoise')
            grp.plot(ax=axForDraw, marker='o', label='initial velocity range = {}'.format(qVelInitNoise))

        plt.ylabel('fraction of trajectories in which the prey is caught')
        plt.legend(loc='best')
        plt.title('heat seeking policy')
        plt.show()

if __name__ == '__main__':
    main()
