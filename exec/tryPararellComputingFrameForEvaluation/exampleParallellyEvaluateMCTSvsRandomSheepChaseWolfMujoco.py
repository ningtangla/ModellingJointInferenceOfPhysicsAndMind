import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
import pickle
import time
import json
import pandas as pd
from matplotlib import pyplot as plt
from collections import OrderedDict
from subprocess import Popen, PIPE

from exec.evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics

class SampleTrajectoriesParallel:
    def __init__(self, codeFileName, numSample):
        self.codeFileName = codeFileName
        self.numSample = numSample

    def __call__(self, oneConditionDf):
        sampleIdStrings = list(map(str, range(self.numSample)))
        indexLevelNames = oneConditionDf.index.names
        oneCondtion = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        oneCondtionString = json.dumps(oneCondtion)
        cmdList = [['python3', self.codeFileName, oneCondtionString, sampleIndex] for sampleIndex in sampleIdStrings]
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        stratTime = time.time()
        for proc in processList:
            stdOut, stdErr = proc.communicate()
            proc.wait()
        processTime = time.time() - stratTime
        print('processTime', processTime)
        return cmdList


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()

    return dataSet


def drawPerformanceLine(dataDf, axForDraw):
    dataDf.plot(ax=axForDraw, y='mean', yerr='std')


def main():
    codeFileName = 'generateTrajactoriesMCTSvsRandomSheepChaseWolfMujoco.py'
    numSample = 20
    sampleTrajectoriesParallel = SampleTrajectoriesParallel(codeFileName, numSample)
    
    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepPolicyName'] = ['mcts', 'random']
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    
    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(sampleTrajectoriesParallel)
    
    # load trajectories and compute statistics on the trajectories
    dirName = os.path.dirname(__file__)
    saveDirectory = os.path.join(dirName, '..', '..', 'data', 'tryParallelComputingFrame', 'trajectory')
    extension = '.pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
    loadTrajectories = LoadTrajectories(getSavePath, loadData)
    computeStatistics = ComputeStatistics(loadTrajectories, measurementFunction = len)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot the statistics
    fig = plt.figure()

    plotCounter = 1

    axForDraw = fig.add_subplot(1, 1, plotCounter)
    drawPerformanceLine(statisticsDf, axForDraw)

    plt.show()
    
    

if __name__ == "__main__":
    main()
