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
from exec.parallelComputing import GenerateTrajectoriesParallel



def main():

    numTrajectories = 42
    # generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleExpLeashedMultiMCTSAgentTraj.py'
    

    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = int(0.75*numCpuCores)
    numCmdList = min(numTrajectories, numCpuToUse)

    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectories, numCmdList)

    startTime = time.time()
    print("start")

    pathParameters = {'agentId': 30}
    cmdList = generateTrajectoriesParallel(pathParameters)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
