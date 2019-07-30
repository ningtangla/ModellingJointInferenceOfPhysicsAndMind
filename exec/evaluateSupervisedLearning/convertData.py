import glob
import os
import sys
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle

def main():
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(DIRNAME, 'tempData')
    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)
    dataSetExtension = '.pickle'
    getSavePath = GetSavePath(dataSetDirectory, dataSetExtension)
    fileNames = glob.glob('tempData/*.pickle')
    trajectories = [loadFromPickle(fileName) for fileName in fileNames]
    __import__('ipdb').set_trace()
    parameters = {'killzoneRadius': 2, 'maxRunningSteps': 25, 'numSimulations': 100, 'sampleIndex': (5000, 9500)}
    trajectorySavePath = getSavePath(parameters)
    saveToPickle(trajectories, trajectorySavePath)

if __name__ == '__main__':
    main()
    
