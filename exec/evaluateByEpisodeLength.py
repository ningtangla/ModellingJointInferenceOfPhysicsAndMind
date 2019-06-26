import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import numpy as np
import pandas as pd
import mcts
import envNoPhysics as env
import wrapperFunctions
import policyValueNet as net
import policies
import os
import subprocess
import play
from collections import OrderedDict
import pickle
import math
from analyticGeometryFunctions import transitePolarToCartesian


def makeVideo(videoName, path):
    absolutePath = os.path.join(os.getcwd(), path)
    os.chdir(absolutePath)
    fps = 5
    crf = 25
    resolution = '1920x1080'
    cmd = 'ffmpeg -r {} -s {} -i %d.png -vcodec libx264 -crf {} -pix_fmt yuv420p {}'.format(fps, resolution, crf, videoName).split(" ")
    subprocess.call(cmd)
    if os.path.exists(videoName):
        [os.remove(file) if file.endswith(".png") else 0 for file in os.listdir(os.getcwd())]
    else:
        print("Demo generate Failed, needs to be done manually")


        # if renderOn:
        #     extendedBound = 10
        #     screen = pg.display.set_mode([xBoundary[1], yBoundary[1] + extendedBound])
        #     render = envRender.Render(screen, self.savePath, 500)
        #     for trajIndex in range(len(demoStates)):
        #         for stepIndex in range(len(demoStates[trajIndex])):
        #             render(demoStates[trajIndex][stepIndex], trajIndex)
        #     if self.saveVideo:
        #         videoName = "mean_{}_{}_trajectories_nn_demo.mp4".format(evalResults['mean'], trajNum)
        #         makeVideo(videoName, self.savePath)


class EvaluateEpisode():
    def __init__(self, sampleTrajectory):
        self.sampleTrajectory = sampleTrajectory

    def __call__(self, df):
        policyDict = df.index.get_level_values('policy')[0]
        policyName, policy = policyDict
        trajNum = df.index.get_level_values('sampleTrajNum')[0]
        maxTrajLen = df.index.get_level_values("maxTrajLen")[0]
        episodes = [self.sampleTrajectory(policy) for cnt in range(trajNum)]
        episodeLength = [len(episode) for episode in episodes]
        avgLen = np.mean(episodeLength)
        stdLen = np.std(episodeLength)
        minLen = np.min(episodeLength)
        maxLen = np.max(episodeLength)
        medLen = np.median(episodeLength)
        return pd.Series({"mean": avgLen, "var": stdLen, "min": minLen, "max": maxLen, "median": medLen})


if __name__ == "__main__":
    sampleTrajNum = 100
    maxTrajLen = 100
    relateDir = "../data/evaluateByEpisodeLength"

    # env
    wolfID = 0
    sheepID = 1
    posIndex = 0
    numOfAgent = 2
    numPosEachAgent = 2
    numStateSpace = numOfAgent * numPosEachAgent
    numActionSpace = 8
    sheepSpeed = 20
    wolfSpeed = sheepSpeed * 0.95
    degrees = [math.pi / 2, 0, math.pi, -math.pi / 2, math.pi / 4, -math.pi * 3 / 4, -math.pi / 4, math.pi * 3 / 4]
    sheepActionSpace = [tuple(np.round(sheepSpeed * transitePolarToCartesian(degree))) for degree in degrees]
    wolfActionSpace = [tuple(np.round(wolfSpeed * transitePolarToCartesian(degree))) for degree in degrees]
    print(sheepActionSpace)
    print(wolfActionSpace)
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    killZoneRadius = 25

    # mcts policy
    getSheepPos = wrapperFunctions.GetAgentPosFromState(sheepID, posIndex, numPosEachAgent)
    getWolfPos = wrapperFunctions.GetAgentPosFromState(wolfID, posIndex, numPosEachAgent)
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    wolfDriectChasingPolicy = policies.HeatSeekingDiscreteDeterministicPolicy(wolfActionSpace, getWolfPos, getSheepPos)
    transition = env.TransitionForMultiAgent(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(np.array(state), [wolfDriectChasingPolicy(state), np.array(action)])
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)

    rewardFunction = lambda state, action: 1

    cInit = 1
    cBase = 1
    calculateScore = mcts.CalculateScore(cInit, cBase)
    selectChild = mcts.SelectChild(calculateScore)

    mctsUniformActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}
    getActionPrior = mctsUniformActionPrior
    initializeChildren = mcts.InitializeChildren(sheepActionSpace, sheepTransition, getActionPrior)
    expand = mcts.Expand(isTerminal, initializeChildren)

    maxRollOutSteps = 5
    rolloutPolicy = lambda state: sheepActionSpace[np.random.choice(range(numActionSpace))]
    heuristic = lambda state: 0
    nodeValue = mcts.RollOut(rolloutPolicy, maxRollOutSteps, sheepTransition, rewardFunction, isTerminal, heuristic)

    numSimulations = 600
    mctsPolicy = mcts.MCTS(numSimulations, selectChild, expand, nodeValue, mcts.backup, mcts.establishSoftmaxActionDist)

    # neuralNetwork Policy
    modelDir = "../data/evaluateNeuralNetwork/savedModels"
    modelName = "60000data_64x4_minibatch_100kIter_contState_actionDist"
    modelPath = os.path.join(modelDir, modelName)
    generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate=0,
                                                       regularizationFactor=0, valueRelativeErrBound=0.0)
    emptyModel = generateModel([64] * 4)
    trainedModel = net.restoreVariables(emptyModel, modelPath)
    nnPolicy = net.ApproximatePolicy(trainedModel, sheepActionSpace)

    reset = lambda: [np.array([10, 10]), np.array([20, 20])]
    sampleTrajectory = play.SampleTrajectoryWithActionDist(maxTrajLen, sheepTransition, isTerminal, reset, play.agentDistToGreedyAction)
    episodes = sampleTrajectory(mctsPolicy)

    independentVariables = OrderedDict()
    independentVariables['policy'] = [('NN',nnPolicy), ('mcts',mctsPolicy)]
    independentVariables['sampleTrajNum'] = [sampleTrajNum]
    independentVariables['maxTrajLen'] = [maxTrajLen]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    MultiIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=MultiIndex)

    evaluateEpisode = EvaluateEpisode(sampleTrajectory)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluateEpisode)

    saveFileName = "mcts_simulation={}_vs_{}.pkl".format(numSimulations, modelName)
    path = os.path.join(relateDir, saveFileName)
    file = open(path, "wb")
    pickle.dump(resultDF, file)
    file.close()