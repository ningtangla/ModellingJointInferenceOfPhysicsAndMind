import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/sheepWolf")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import numpy as np
import mcts
import noPhysicsEnv as env
import envSheepChaseWolf
import policyValueNet as net
import policiesFixed
import os
import subprocess
import play


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

if __name__ == "__main__":
    evaluateTrajNum = 100
    maxTrajLen = 100

    # env
    wolfID = 0
    sheepID = 1
    posIndex = 0
    numOfAgent = 2
    numPosEachAgent = 2
    numStateSpace = numOfAgent * numPosEachAgent
    actionSpace = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    numActionSpace = len(actionSpace)
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    sheepVelocity = 20
    killZoneRadius = 25
    wolfVelocity = sheepVelocity * 0.95

    # mcts policy
    getSheepPos = envSheepChaseWolf.GetAgentPos(sheepID, posIndex, numPosEachAgent)
    getWolfPos = envSheepChaseWolf.GetAgentPos(wolfID, posIndex, numPosEachAgent)
    checkBoundaryAndAdjust = env.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    wolfDriectChasingPolicy = policiesFixed.PolicyActionDirectlyTowardsOtherAgent(getWolfPos, getSheepPos, wolfVelocity)
    transition = env.TransitionForMultiAgent(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(np.array(state), [np.array(action), wolfDriectChasingPolicy(state)])
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)

    rewardFunction = lambda state, action: 1

    cInit = 1
    cBase = 1
    calculateScore = mcts.CalculateScore(cInit, cBase)
    selectChild = mcts.SelectChild(calculateScore)

    getActionPrior = mcts.GetActionPrior(actionSpace)
    initializeChildren = mcts.InitializeChildren(actionSpace, sheepTransition, getActionPrior)
    expand = mcts.Expand(isTerminal, initializeChildren)

    maxRollOutSteps = 5
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    heuristic = mcts.HeuristicDistanceToTarget(1, getSheepPos, getWolfPos)
    nodeValue = mcts.RollOut(rolloutPolicy, maxRollOutSteps, sheepTransition, rewardFunction, isTerminal, heuristic)

    numSimulations = 600
    mctsPolicy = mcts.MCTS(numSimulations, selectChild, expand, nodeValue, mcts.backup, mcts.getGreedyAction)

    reset = lambda: [np.array([10, 10]), np.array([90, 90])]
    sampleTrajectory = play.SampleTrajectory(maxTrajLen, transition, isTerminal, reset)
    episodes = sampleTrajectory(mctsPolicy)
    print(episodes)
