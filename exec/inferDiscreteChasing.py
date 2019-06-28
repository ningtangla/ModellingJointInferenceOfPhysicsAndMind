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
    actHeatSeeking = ActHeatSeeking(actionSpace, computeAngleBetweenVectors, lowerBoundAngle, upperBoundAngle)

    positionIndex = [0, 1]
    rationalityParam = 0.9
    actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1), (0, 0)]

    getAgentPosition = lambda agentID: GetAgentPosFromState(agentID, positionIndex)

    getHeatSeekingActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam,actHeatSeeking, getWolfPos, getSheepPos)
    getMasterActionLikelihood = RandomActionLikelihood(actionSpace)

    getPolicyDistribution = GetPolicyDistribution(getAgentPosition, getHeatSeekingActionLikelihood, getMasterActionLikelihood)

    inferPolicyLikelihood = InferPolicyLikelihood(getPolicyDistribution)



    getPulledAgentForceLikelihood = PulledForceDirectionLikelihood(computeAngleBetweenVectors, forceSpace, lowerBoundAngle, upperBoundAngle)

    getPulledAgentForceDistribution = GetPulledAgentForceDistribution(getAgentPosition, getPulledAgentForceLikelihood)

    inferForceLikelihood = InferForceLikelihood(getPulledAgentForceDistribution)

    inferOneStepDiscreteChasing = InferOneStepDiscreteChasing(chasingAgents, pullingAgents,
                                                              actionSpace, forceSpace, createIndex,
                                                              inferPolicyLikelihood,
                                                              inferForceLikelihood,
                                                              inferTransitionLikelihood)

    state = [(2, 2), (3, 3), (4, 5)]
    nextState = [(2, 3), (1, 3), (4, 5)]
    inferenceDf = inferOneStepDiscreteChasing(state, nextState)

    inferenceDf.to_csv("oneStepResult.csv")

if __name__ == '__main__':
    main()














