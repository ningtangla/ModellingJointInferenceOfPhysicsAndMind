import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import numpy as np

from src.constrainedChasingEscapingEnv.envMujoco import WithinBounds, ResetUniform
from exec.trainMCTSNNIteratively.resetGaussian import ResetGaussian

class PlotWolfStartStates:
    def __init__(self, getResetGaussian, numSamples, getWolfQPosFromState, getSheepQPosFromState):
        self.getResetGaussian = getResetGaussian
        self.numSamples = numSamples
        self.getWolfQPosFromState = getWolfQPosFromState
        self.getSheepQPosFromState = getSheepQPosFromState

    def __call__(self, oneConditionDf, axForDraw):
        initStDev = oneConditionDf.index.get_level_values('initStDev')[0]
        resetGaussian = self.getResetGaussian(initStDev)
        allStartStates = [resetGaussian() for _ in range(self.numSamples)]
        allQPosWolf = [self.getWolfQPosFromState(state) for state in allStartStates]
        allQPosSheep = [self.getSheepQPosFromState(state) for state in allStartStates]
        # allQPosWolf = np.random.uniform(-9.7, 9.7, (self.numSamples, 2))
        # allQPosSheep = np.random.uniform(-9.7, 9.7, (self.numSamples, 2))
        axForDraw.scatter(*zip(*allQPosWolf))
        axForDraw.scatter(*zip(*allQPosSheep))
        axForDraw.set_title("StdDev: {}".format(initStDev))

        return None


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['initStDev'] = [16]
    numSamples = 5000

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    numAgents = 2
    minQPos = (-9.7, -9.7)
    maxQPos = (9.7, 9.7)
    withinBounds = WithinBounds(minQPos, maxQPos)
    qVelInit = (0, 0, 0, 0)
    qVelInitStDev = 1
    qMin = (-0.1, -0.1)
    qMax = (0, 0)
    getResetGaussian = lambda qPosInitStDev: ResetGaussian(physicsSimulation, qVelInit, numAgents, qPosInitStDev,
                                                           qVelInitStDev, qMin, qMax, withinBounds)
    getWolfQPosFromState = lambda state: state[1:, :2].flatten().tolist()
    getSheepQPosFromState = lambda state: state[:1, :2].flatten().tolist()

    # plot
    plotStartStates = PlotWolfStartStates(getResetGaussian, numSamples, getWolfQPosFromState, getSheepQPosFromState)

    fig = plt.figure()
    numRows = 1
    numColumns = len(manipulatedVariables['initStDev'])
    pltCounter = 1

    for std, grp in toSplitFrame.groupby('initStDev'):
        axForDraw = fig.add_subplot(numRows, numColumns, pltCounter)
        axForDraw.set_xlim(-10, 10)
        axForDraw.set_ylim(-10, 10)
        plotStartStates(grp, axForDraw)
        pltCounter += 1

    plt.show()


if __name__ == '__main__':
    main()