import sys

sys.path.append('../src/inferDiscreteGridChasing')
sys.path.append('../src/constrainedChasingEscapingEnv')

from forceLikelihood import *
from heatSeekingLikelihood import *
from analyticGeometryFunctions import computeAngleBetweenVectors
from inference import *
from wrapperFunctions import *

pd.set_option('display.max_columns', 50)


def main():
    chasingAgents = [0, 1, 2]
    pullingAgents = [0, 1, 2]
    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
    forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]

    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2
    getWolfProperAction = ActHeatSeeking(actionSpace, computeAngleBetweenVectors, lowerBoundAngle, upperBoundAngle)
    getSheepProperAction = ActHeatSeeking(actionSpace, computeAngleBetweenVectors, lowerBoundAngle, upperBoundAngle)

    positionIndex = [0, 1]
    rationalityParam = 0.9
    actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1), (0, 0)]

    getWolfPosition = lambda wolfID: GetAgentPosFromState(wolfID, positionIndex)
    getSheepPosition = lambda sheepID: GetAgentPosFromState(sheepID, positionIndex)


    getWolfActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam, getWolfProperAction, getWolfPos, getSheepPos)
    getSheepActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam, getSheepProperAction, getWolfPos, getSheepPos)
    getMasterActionLikelihood = RandomActionLikelihood(actionSpace)


    inferPolicyLikelihood = InferPolicyLikelihood(positionIndex, rationalityParam, actionSpace,
                                                  getWolfPosition, getSheepPosition,
                                                  getWolfActionLikelihood, getSheepActionLikelihood,
                                                  getMasterActionLikelihood)

    getPulledAgentPos = lambda pulledAgentID: GetAgentPosFromState(pulledAgentID, positionIndex)
    getPullingAgentPos = lambda pullingAgentID: GetAgentPosFromState(pullingAgentID, positionIndex)

    getPulledAgentForceLikelihood = PulledForceDirectionLikelihood(computeAngleBetweenVectors, forceSpace,
                                                                   lowerBoundAngle, upperBoundAngle)

    inferForceLikelihood = InferForceLikelihood(positionIndex, forceSpace,
                                                getPulledAgentPos, getPullingAgentPos,
                                                getPulledAgentForceLikelihood)

    inferOneStepDiscreteChasing = InferOneStepDiscreteChasing(chasingAgents, pullingAgents,
                                                              actionSpace, forceSpace, createIndex,
                                                              inferPolicyLikelihood,
                                                              inferForceLikelihood,
                                                              inferTransitionLikelihood)

    state = [(2, 2), (3, 3), (4, 5)]
    nextState = [(2, 3), (1, 3), (4, 5)]
    inferenceDf = inferOneStepDiscreteChasing(state, nextState)

    #inferenceDf.to_excel("df.xlsx")

    print(inferenceDf.head())

    #state = [(2,2), (3,3), (4,5)]

    # allAgentsActions = ((0, 1), (-1, 0), (-1, 0))
    # chasingIndices = (1, 0, 2) # ('sheep', 'wolf', 'master')
    # print(inferPolicyLikelihood(state, allAgentsActions, chasingIndices))

    # pullingIndices = (2, 1, 0) #  pulling,  noPull, pulled
    # allAgentsForces = ((1, 0), (0, 0), (-1, 0))
    #
    # print(inferForceLikelihood(state, allAgentsForces, pullingIndices))
    #
    # observedNextState =
    #
    # print(inferTransitionLikelihood(state, allAgentsActions, allAgentsForces, observedNextState))
    # #?? reach the boundary

if __name__ == '__main__':
    main()














