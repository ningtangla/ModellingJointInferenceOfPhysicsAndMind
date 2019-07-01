import sys

sys.path.append('../src/inferDiscreteGridChasing')
sys.path.append('../src/constrainedChasingEscapingEnv')

from forceLikelihood import *
from heatSeekingLikelihood import *
from analyticGeometryFunctions import computeAngleBetweenVectors
from inference import *
from wrapperFunctions import *
from wrappers import *

pd.set_option('display.max_columns', 50)


def main():
    chasingAgents = [0, 1, 2]
    pullingAgents = [0, 1, 2]
    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]

    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2
    actHeatSeeking = ActHeatSeeking(actionSpace, computeAngleBetweenVectors, lowerBoundAngle, upperBoundAngle)

    positionIndex = [0, 1]
    rationalityParam = 0.9

    getAgentPosition = lambda agentID: GetAgentPosFromState(agentID, positionIndex)

    getHeatSeekingActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam,actHeatSeeking, getWolfPos, getSheepPos)
    getMasterActionLikelihood = RandomActionLikelihood(actionSpace)

    getPolicyDistribution = GetPolicyDistribution(getAgentPosition, getHeatSeekingActionLikelihood, getMasterActionLikelihood)

    inferPolicyLikelihood = InferPolicyLikelihood(getPolicyDistribution)


    getPulledAgentForceLikelihood = PulledForceDirectionLikelihood(computeAngleBetweenVectors, forceSpace, lowerBoundAngle, upperBoundAngle)

    getPulledAgentForceDistribution = GetPulledAgentForceDistribution(getAgentPosition, getPulledAgentForceLikelihood)

    inferForceLikelihood = InferForceLikelihood(getPulledAgentForceDistribution)

    inferOneStepDiscreteChasing = InferOneStepDiscreteChasing(inferPolicyLikelihood, inferForceLikelihood, inferTransitionLikelihood)

    isTerminal = IsTerminal(0.4, ['chasingAgents', 'pullingAgents'])

    inferDiscreteChasing = InferDiscreteChasing(chasingAgents, pullingAgents, actionSpace, forceSpace,
                 createIndex, isTerminal, inferOneStepDiscreteChasing)

    trajectory = [[(7, 3), (5, 4), (4, 8)],
                  [(6, 4), (4, 4), (4, 8)],
                  [(4, 4), (3, 4), (5, 7)],
                  [(3, 5), (2, 4), (6, 6)],
                  [(3, 5), (2, 3), (6, 6)],
                  [(4, 4), (2, 2), (5, 7)],
                  [(5, 3), (2, 3), (4, 8)]]

#
    inferenceDf = inferDiscreteChasing(trajectory)

    inferenceDf.to_csv("chasingInference.csv")


if __name__ == '__main__':
    main()





