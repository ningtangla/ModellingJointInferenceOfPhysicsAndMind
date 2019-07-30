import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
from collections import OrderedDict
import pandas as pd

from exec.parallelComputing import GenerateTrajectoriesParallel


from subprocess import Popen, PIPE
import json
import math

class TrainMultiMCTSAgentParallel:
    def __init__(self, codeFileName):
        self.codeFileName = codeFileName

    def __call__(self, hyperParameterConditionslist):
        [print(condition) for condition in hyperParameterConditionslist]
        cmdList = [['python3', self.codeFileName, json.dumps(condition)]
                for condition in hyperParameterConditionslist]

        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.wait()
        return cmdList

def main():
    # Mujoco environment
    print('start')
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
        oneCondition={levelName:str(modelIndex.get_level_values(levelName)[modelIndexNumber]) for levelName in levelNames}
        hyperVariablesConditionlist.append(oneCondition)
    print(hyperVariablesConditionlist)
    numTrajectoriesToStartTrain = 4 * 256

    #generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleMCTSWolfTrajectory.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList)

    print('StratParallelGenerate')
    for numSimulations in  manipulatedHyperVariables['numSimulations']:
        trajectoryBeforeTrainPathParamters = {'iterationIndex': 0,'numSimulations':numSimulations}
        preTrainCmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)
        print(preTrainCmdList)


    trainOneConditionFileName='trainMultiMCTSforOneCondition.py'

    trainMultiMCTSAgentParallel=TrainMultiMCTSAgentParallel(trainOneConditionFileName)

    trainCmdList=trainMultiMCTSAgentParallel(hyperVariablesConditionlist)

    print(trainCmdList)



if __name__ == '__main__':
    main()
