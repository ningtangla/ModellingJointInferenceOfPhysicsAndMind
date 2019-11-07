import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))
# import ipdb

import numpy as np
from collections import OrderedDict

from exec.parallelComputing import GenerateTrajectoriesParallel


def main():
    dirName = os.path.dirname(__file__)

    wolfId = 1

    startTime = time.time()
    numTrajectories = 3000
    # generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleMCTSCenterControlWovles.py'

    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = int(0.75*numCpuCores)
    numCmdList = min(numTrajectories, numCpuToUse)

    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectories, numCmdList)


    print("start")
    trainableAgentIds = [wolfId]
    for agentId in trainableAgentIds:
        print("agent {}".format(agentId))
        pathParameters = {'agentId': agentId}

        cmdList = generateTrajectoriesParallel(pathParameters)
        print(cmdList)
        # trajectories = loadTrajectoriesForParallel(pathParameters)
        # import ipdb; ipdb.set_trace()

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
