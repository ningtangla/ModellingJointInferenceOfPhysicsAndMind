import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..'))
# import ipdb

import numpy as np
from collections import OrderedDict, deque
import pandas as pd
import mujoco_py as mujoco
import itertools as it

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import  SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel, ExcuteCodeOnConditionsParallel



def main():
    startTime = time.time()

    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['tendonStiffness'] = [10]
    manipulatedVariables['tendonDamping'] =  [0.7]
    manipulatedVariables['maxTendonLength'] = [0.6]
    manipulatedVariables['predatorMass'] = [10]
    manipulatedVariables['draggerMass'] = [8, 10, 12]
    manipulatedVariables['predatorPower'] = [1, 1.3, 1.6]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleMCTSLeasedWolfTrajectory.py'

    generateTrajectoriesParallel = ExcuteCodeOnConditionsParallel(sampleTrajectoryFileName)

    print("start")
    cmdList = generateTrajectoriesParallel(parametersAllCondtion)
    # import ipdb; ipdb.set_trace()

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
