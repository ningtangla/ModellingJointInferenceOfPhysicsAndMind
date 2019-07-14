import sys
sys.path.append("..")
sys.path.append("../../src/neuralNetwork")
sys.path.append("../../src/constrainedChasingEscapingEnv")
sys.path.append("../../src/algorithms")
sys.path.append("../../src")
import numpy as np
from collections import Counter
import pickle
from evaluationFunctions import GetSavePath
from measurementFunctions import calculateCrossEntropy


def multiDistCrossEntropy(dists):
    dists = np.array([np.array(dist) for dist in dists])
    meanDist = np.mean(dists, axis=0)
    entropies = np.array([calculateCrossEntropy({"d1": meanDist, "d2": dist}) for dist in dists])
    sumEntropy = np.sum(entropies)
    return sumEntropy


def inspectDataSet():
    extension = ".pickle"
    dataSetsDir = '../../data/compareValueDataStandardizationAndLossCoefs/trainingData/dataSets'
    getDataSetPath = GetSavePath(dataSetsDir, extension)

    initPosition = np.array([[30, 30], [20, 20]])
    maxRollOutSteps = 10
    numSimulations = 200
    maxRunningSteps = 30
    numTrajs = 1000
    numDataPoints = 29000
    # numTrajs = 200
    # numDataPoints = 5800
    useStdReward = True
    cBase = 100
    varDict = {}
    varDict["initPos"] = list(initPosition.flatten())
    varDict["rolloutSteps"] = maxRollOutSteps
    varDict["numSimulations"] = numSimulations
    varDict["maxRunningSteps"] = maxRunningSteps
    varDict["numTrajs"] = numTrajs
    varDict["numDataPoints"] = numDataPoints
    varDict["standardizedReward"] = useStdReward
    varDict["cBase"] = cBase
    savePath = getDataSetPath(varDict)

    dataSize = 3000
    with open(savePath, "rb") as f:
        dataSet = pickle.load(f)
    dataPoints = list(zip(*list(dataSet.values())))[:dataSize]

    counter = Counter()
    for s, a, ad, v in dataPoints:
        sTuple = tuple(s)
        counter[sTuple] += 1

    reps = np.array(list(counter.values()))
    meanRep = np.mean(reps)
    stdRep = np.std(reps)
    minRep = np.min(reps)
    medRep = np.median(reps)
    maxRep = np.max(reps)
    print("mean {} std {} min {} med {} max {}".format(meanRep, stdRep, minRep, medRep, maxRep))

    k = 15
    print("----Top {} most common states:".format(k))
    for state, rep in counter.most_common(k):
        print("State: {}, rep: {}".format(state, rep))

    stateSumEntropies = []
    for state in counter:
        actionDists = []
        for s, a, ad, v in dataPoints:
            if tuple(s) == state:
                actionDists.append(ad)
        stateSumEntropies.append(multiDistCrossEntropy(actionDists))
    assert(len(stateSumEntropies) == len(counter))
    assert(np.sum(reps) == dataSize)
    print(np.sum(stateSumEntropies) / np.sum(reps))


if __name__ == "__main__":
    inspectDataSet()
