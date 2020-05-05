import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from subprocess import Popen, PIPE
import json
import math
import numpy as np
from collections import OrderedDict
import pathos.multiprocessing as mp
import itertools as it

class GenerateTrajectoriesParallel:
    def __init__(self, codeFileName):
        self.codeFileName = codeFileName

    def __call__(self, startEndIndexesPair, parameters):
        parametersString = dict([(key, str(value)) for key, value in parameters.items()])
        parametersStringJS = json.dumps(parametersString)
        cmdList = [['python3', self.codeFileName, parametersStringJS, str(startSampleIndex), str(endSampleIndex)] for startSampleIndex, endSampleIndex in startEndIndexesPair]
        print(cmdList)
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.communicate()
        return cmdList


def sampleMCTSOneCondtion(parameters):


    numSimulations=parameters['numSimulations']
    maxRolloutSteps=parameters['maxRolloutSteps']


    dirName = os.path.dirname(__file__)

    wolfId = 1
    pathParameters = {'agentId': wolfId}
    parameters.update(pathParameters)

    startTime = time.time()
    numTrajectories = 500
    # generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'generateMCTSTrajcectory.py'

    numCpuCores = os.cpu_count()
    numCpuToUse = int(2)
    numCmdList = min(numTrajectories, numCpuToUse)
    print('numCpuToUse',numCpuToUse)

    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName)

    numTrajPerSteps = numCmdList * 20
    startSampleIndexes = np.arange(0, numTrajectories, math.ceil(numTrajPerSteps / numCmdList))
    endSampleIndexes = np.concatenate([startSampleIndexes[1:], [numTrajectories]])
    startEndIndexesPairs = list(zip(startSampleIndexes, endSampleIndexes))

    print("start")
    for i in range(math.ceil(numTrajectories / numTrajPerSteps)):
        startTime = time.time()

        startEndIndexesPair = startEndIndexesPairs[numCmdList * i : numCmdList * i + numCmdList]
        cmdList = generateTrajectoriesParallel(startEndIndexesPair, parameters)

        endTime = time.time()
        print("Time taken {} seconds".format((endTime - startTime)))

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [50,100,150,200]
    manipulatedVariables['maxRolloutSteps'] = [10,20,30]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    #parallel train
    numCpuCores = os.cpu_count()
    numCpuToUse = 6
    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(sampleMCTSOneCondtion,parametersAllCondtion)

    #evaluate
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    dirName = os.path.dirname(__file__)
    dataFolderName=os.path.join(dirName,'..','..', '..', 'data', 'multiAgentTrain', 'MCTSFixObstacle')
    trajectoryDirectory = os.path.join(dataFolderName, 'trajectories')

    trajectoryExtension = '.pickle'
    maxRunningSteps = 30
    killzoneRadius = 2
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius,'agentId':agentId}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # compute statistics on the trajectories
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))

    decay = 1
    accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)
    measurementFunction = lambda trajectory: accumulateMultiAgentRewards(trajectory)[0]

    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)


if __name__ == '__main__':
    main()
