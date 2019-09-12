import sys
import os
sys.path.append(os.path.join('..'))

import itertools as it
import pandas as pd
import numpy as np
from pygame.color import THECOLORS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from src.constrainedChasingEscapingEnv.envDiscreteGrid import Reset, StayWithinBoundary, GetPullingForceValue, \
    GetPulledAgentForce, SamplePulledForceDirection, GetAgentsForce, Transition, IsTerminal
from src.episode import MultiAgentSampleTrajectory


from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.policies import RandomPolicy, ActHeatSeeking, HeatSeekingDiscreteStochasticPolicy, RandomPolicyChooseAction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState

from src.inferChasing.discreteGridPolicy import ActHeatSeeking, \
    HeatSeekingPolicy, WolfPolicy, SheepPolicy, MasterPolicy
from src.inferChasing.discreteGridTransition import StayWithinBoundary, PulledForceLikelihood, \
    PulledTransition, NoPullTransition
from src.inferChasing.inference import IsInferenceTerminal, ObserveStateOnly, InferOneStep, \
    InferDiscreteChasingAndDrawDemo

from visualize.inferenceVisualization import SaveImage, GetChasingRoleColor, \
    GetChasingResultColor, ColorChasingPoints, AdjustPullingLineWidth, \
    DrawInferenceResultWithPull, PlotInferenceProb
from visualize.initialization import initializeScreen
from visualize.discreteGridVisualization import checkDuplicates, ModifyOverlappingPoints, DrawGrid,\
    DrawCirclesAndLines

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    wolfActionSpace = [(-2, 0), (2, 0), (0, 2), (0, -2)]
    sheepActionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    randomActionSpace = sheepActionSpace

    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2

    wolfActHeatSeeking = ActHeatSeeking(wolfActionSpace, lowerBoundAngle, upperBoundAngle, computeAngleBetweenVectors)
    sheepActHeatSeeking = ActHeatSeeking(sheepActionSpace, lowerBoundAngle, upperBoundAngle, computeAngleBetweenVectors)

    rationalityParam = 0.9
    wolfHeatSeekingPolicy = HeatSeekingPolicy(rationalityParam, wolfActHeatSeeking)
    sheepHeatSeekingPolicy = HeatSeekingPolicy(rationalityParam, sheepActHeatSeeking)

    gridSize = (10,10)
    iterationNumber = 10
    agentNames = ["Wolf", "Sheep", "Master"]

    wolfID = 0
    sheepID = 1
    masterID = 2
    positionIndex = [0, 1] # [2, 0, 1] state = [sheepState, masterState, wolfState]

    getWolfPos = GetAgentPosFromState(wolfID, positionIndex)
    getSheepPos = GetAgentPosFromState(sheepID, positionIndex)

    wolfPolicy = HeatSeekingDiscreteStochasticPolicy(rationalityParam, wolfActHeatSeeking, getWolfPos, getSheepPos)
    sheepPolicy = HeatSeekingDiscreteStochasticPolicy(rationalityParam, sheepActHeatSeeking, getWolfPos, getSheepPos)
    masterPolicy = RandomPolicyChooseAction(randomActionSpace)

    unorderedPolicy = [wolfPolicy, sheepPolicy, masterPolicy]
    agentID = [wolfID, sheepID, masterID]

    rearrangeList = lambda unorderedList, order: list(np.array(unorderedList)[np.array(order).argsort()])
    allAgentPolicy = rearrangeList(unorderedPolicy, agentID) # sheepPolicy, masterPolicy, wolfPolicy

    policy = lambda state: [getAction(state) for getAction in allAgentPolicy] # result of policy function is [sheepAct, masterAct, wolfAct]

    pulledAgentID = 0
    noPullAgentID = 1
    pullingAgentID = 2

    getPulledAgentPos = GetAgentPosFromState(pulledAgentID, positionIndex)
    getPullingAgentPos = GetAgentPosFromState(pullingAgentID, positionIndex)
    lowerGridBound = 1
    agentCount = 3
    reset = Reset(gridSize, lowerGridBound, agentCount)
    stayWithinBoundary = StayWithinBoundary(gridSize, lowerGridBound)

    forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
    distanceForceRatio = 20
    getPullingForceValue = GetPullingForceValue(distanceForceRatio)
    samplePulledForceDirection = SamplePulledForceDirection(computeAngleBetweenVectors, forceSpace, lowerBoundAngle, upperBoundAngle)

    getPulledAgentForce = GetPulledAgentForce(getPullingAgentPos, getPulledAgentPos, samplePulledForceDirection, getPullingForceValue)
    getAgentsForce = GetAgentsForce(getPulledAgentForce, pulledAgentID, noPullAgentID, pullingAgentID)
    transition = Transition(stayWithinBoundary, getAgentsForce)
    isTerminal = IsTerminal(getWolfPos, getSheepPos)

    getMultiAgentSampleTrajectory = MultiAgentSampleTrajectory(agentNames, iterationNumber, isTerminal, reset)
    trajectory = getMultiAgentSampleTrajectory(policy, transition)
    print(trajectory)

    ##############################
    positionIndex = [0, 1]
    getAgentPosition = lambda agentID, state: GetAgentPosFromState(agentID, positionIndex)(state)

    wolfPolicy = WolfPolicy(getAgentPosition, wolfHeatSeekingPolicy)
    sheepPolicy = SheepPolicy(getAgentPosition, sheepHeatSeekingPolicy)

    uniformPolicy = RandomPolicy(randomActionSpace)
    masterPolicy = MasterPolicy(uniformPolicy)

    policyList = [wolfPolicy, sheepPolicy, masterPolicy]
    policy = lambda mind, state, allAgentsAction: np.product(
        [agentPolicy(mind, state, allAgentsAction) for agentPolicy in policyList])


    forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
    getPulledAgentForceLikelihood = PulledForceLikelihood(forceSpace, lowerBoundAngle,
                                                          upperBoundAngle, computeAngleBetweenVectors)
    pulledTransition = PulledTransition(getAgentPosition, getPulledAgentForceLikelihood,
                                        stayWithinBoundary)
    noPullTransition = NoPullTransition(getAgentPosition, stayWithinBoundary)
    transitionList = [pulledTransition, noPullTransition]
    transition = lambda physics, state, allAgentsAction, nextState: \
        np.product([agentTransition(physics, state, allAgentsAction, nextState)
                    for agentTransition in transitionList])

    getMindsPhysicsActionsJointLikelihood = lambda mind, state, allAgentsAction, physics, nextState: \
        policy(mind, state, allAgentsAction) * transition(physics, state, allAgentsAction, nextState)


# visualization
    fullScreen = False
    screenWidth = 800
    screenHeight = 800
    screen = initializeScreen(fullScreen, screenWidth, screenHeight)

    gridNumberX, gridNumberY = gridSize
    gridPixelSize = min(screenHeight// gridNumberX, screenWidth// gridNumberY)
    modificationRatio = 3
    modifyOverlappingPoints = ModifyOverlappingPoints(gridPixelSize, checkDuplicates, modificationRatio)
    drawCirclesAndLines = DrawCirclesAndLines(modifyOverlappingPoints)
    drawGrid = DrawGrid(screen, gridSize, gridPixelSize)

    wolfColor = THECOLORS['red']
    sheepColor = THECOLORS['green']
    masterColor = THECOLORS['blue']

    wolfIndex = 'wolf'
    sheepIndex = 'sheep'
    masterIndex = 'master'

    getWolfColor = GetChasingRoleColor(wolfColor, wolfIndex)
    getSheepColor = GetChasingRoleColor(sheepColor, sheepIndex)
    getMasterColor = GetChasingRoleColor(masterColor, masterIndex)
    getRolesColor = [getWolfColor, getSheepColor, getMasterColor]
    getChasingResultColor = GetChasingResultColor(getRolesColor)

    colorChasingPoints = ColorChasingPoints(getChasingResultColor)
    adjustPullingLineWidth = AdjustPullingLineWidth()

    chasingAgents = ['wolf', 'sheep', 'master']
    chasingSpace = list(it.permutations(chasingAgents))
    pullingAgents = ['pulled', 'noPull', 'pulled']
    pullingSpaceArray = np.unique(list(it.permutations(pullingAgents)), axis=0)
    pullingSpace = [tuple(pullingPair) for pullingPair in pullingSpaceArray.tolist()]

    actionHypoList = [list(it.permutations([x,y,z])) for x in wolfActionSpace for y in sheepActionSpace for z in randomActionSpace]
    actionHypoWithDuplicates = list(it.chain.from_iterable(actionHypoList))
    actionHypo = [tuple([tuple(action) for action in actionList]) for actionList in
                  np.unique(actionHypoWithDuplicates, axis=0).tolist()]

    iterables = [chasingSpace, pullingSpace, actionHypo]
    inferenceIndex = pd.MultiIndex.from_product(iterables, names=['mind', 'physics', 'action'])

    drawInferenceResult = DrawInferenceResultWithPull(inferenceIndex, gridPixelSize,
                                                      drawGrid, drawCirclesAndLines,
                                                      colorChasingPoints, adjustPullingLineWidth)

    thresholdPosterior = 1.5
    mindPhysicsName = ['mind', 'physics']
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, mindPhysicsName, inferenceIndex)

    # inferOneStepLikelihood = InferOneStepLikelihood(inferenceIndex, getMindsPhysicsActionsJointLikelihood)
    inferOneStep = InferOneStep(inferenceIndex, mindPhysicsName, getMindsPhysicsActionsJointLikelihood)

    observe = ObserveStateOnly(trajectory)

    dataIndex = 0
    imageFolderName = 'demo' + str(dataIndex)
    saveImage = SaveImage(imageFolderName)
    FPS = 60

    inferDiscreteChasingAndDrawDemo = InferDiscreteChasingAndDrawDemo(FPS, inferenceIndex, isInferenceTerminal,
                 observe, inferOneStep, drawInferenceResult, saveImage)


    mindsPhysicsPrior = [1/ len(inferenceIndex)] * len(inferenceIndex)

    posteriorDf = inferDiscreteChasingAndDrawDemo(mindsPhysicsPrior)

    plotMindInferenceProb = PlotInferenceProb('timeStep', 'mindPosterior', 'mind')
    plotPhysicsInferenceProb = PlotInferenceProb('timeStep', 'physicsPosterior', 'physics')

    plotName = 'Discrete3AgentsPullInference_ModifyActionHypo'
    plotMindInferenceProb(posteriorDf, dataIndex, plotName)
    plotPhysicsInferenceProb(posteriorDf, dataIndex, plotName)



if __name__ == '__main__':
    main()
