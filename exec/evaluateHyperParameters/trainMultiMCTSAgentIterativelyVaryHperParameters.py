import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
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
from src.episode import SampleTrajectory, sampleAction
from exec.parallelComputing import GenerateTrajectoriesParallel


from subprocess import Popen, PIPE
import json
import math

class TrainMultiMCTSAgentParallel:
    def __init__(self, codeFileName, numSample, numCmdList):
        self.codeFileName = codeFileName
        self.numSample = numSample
        self.numCmdList = numCmdList
    def __call__(self, hyperParameterConditionslist):

        cmdList = [['python3', self.codeFileName, json.dumps(condition)]
                for condition in hyperParameterConditionslist]
        print(cmdList)
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.wait()
        return cmdList

def main():
    # Mujoco environment
    manipulatedHyperVariables = OrderedDict()
    manipulatedHyperVariables['miniBatchSize'] = [64, 256]  # [64, 128, 256]
    manipulatedHyperVariables['learningRate'] = [1e-3, 1e-4, 1e-5]  # [1e-2, 1e-3, 1e-4]
    manipulatedHyperVariables['numSimulations'] = [50] #[50, 100, 200]

    #numSimulations = manipulatedHyperVariables['numSimulations']
    levelNames = list(manipulatedHyperVariables.keys())
    levelValues = list(manipulatedHyperVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

    hyperVariablesConditionlist=[]

    for modelIndexNumber in range(len(modelIndex)):
        oneCondition={levelName:modelIndex.get_level_values(levelName)[modelIndexNumber] for levelName in levelNames}
        hyperVariablesConditionlist.append(oneCondition)

    numTrajectoriesToStartTrain = 4 * 256

    #generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleMultiMCTSAgentTrajectory.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList, readParametersFromDf)

    for numSimulations in  manipulatedHyperVariables['numSimulations']:
        trajectoryBeforeTrainPathParamters = {'iterationIndex': 0,'numSimulations':numSimulations}
        preTrainCmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)
    print(preTrainCmdList)


    trainOneConditionFileName='trainMultiMCTSforOneCondition.py'
    trainMultiMCTSAgentParallel=TrainMultiMCTSAgentParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList, readParametersFromDf)

    trainCmdList=trainMultiMCTSAgentParallel(hyperVariablesConditionlist)
    print(trainCmdList)



if __name__ == '__main__':
    main()
