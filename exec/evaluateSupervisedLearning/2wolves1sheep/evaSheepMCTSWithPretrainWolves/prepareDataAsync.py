import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))
import numpy as np
from collections import OrderedDict

from subprocess import Popen, PIPE
import json
import math
import numpy as np
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


def main():
    dirName = os.path.dirname(__file__)

    sheepId = 0
    numSimulationsList = [110, 150, 200]
    maxRolloutStepsList = [10, 15, 20]

    startTime = time.time()
    numTrajectories = 300
    sampleTrajectoryFileName = 'sampleMCTSSheepMultiChasingNoPhysics.py'

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    numCmdList = min(numTrajectories, numCpuToUse)
    print('numCpuToUse', numCpuToUse)

    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName)

    numTrajPerSteps = numCmdList * 2
    startSampleIndexes = np.arange(0, numTrajectories, math.ceil(numTrajPerSteps / numCmdList))
    endSampleIndexes = np.concatenate([startSampleIndexes[1:], [numTrajectories]])
    startEndIndexesPairs = list(zip(startSampleIndexes, endSampleIndexes))

    for numSimulations, maxRolloutSteps in it.product(numSimulationsList, maxRolloutStepsList):
        pathParameters = {'agentId': sheepId, 'numSimulations': numSimulations, 'maxRolloutSteps': maxRolloutSteps}

        print("start", startTime)
        for i in range(math.ceil(numTrajectories / numTrajPerSteps)):
            startTime = time.time()

            startEndIndexesPair = startEndIndexesPairs[numCmdList * i: numCmdList * i + numCmdList]
            cmdList = generateTrajectoriesParallel(startEndIndexesPair, pathParameters)

            endTime = time.time()
        print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
