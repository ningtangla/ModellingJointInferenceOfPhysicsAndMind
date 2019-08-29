import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
import pandas as pd
import mujoco_py as mujoco
import itertools as it

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionAvoidCollisionAndWall, IsCollided,HeuristicDistanceToTarget
from exec.trajectoriesSaveLoad import loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, StochasticMCTS, backup, establishPlainActionDistFromMultipleTrees, RollOut
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import RandomPolicy
from src.episode import SampleAction

from src.inferChasing.continuousPolicy import InferencePolicy
from src.inferChasing.continuousTransition import TransitConstantPhysics
from src.inferChasing.inference import IsInferenceTerminal, Observe, QueryDecayedLikelihood, \
    InferOneStepLikelihood, InferContinuousChasingAndDrawDemo, softenPolicy
from visualize.inferenceVisualization import PlotInferenceProb



class ApproximatePolicy:
    def __init__ (self, policyValueNet, actionSpace, agentStateIdsForNN):
        self.policyValueNet = policyValueNet
        self.actionSpace = actionSpace
        self.agentStateIdsForNN = agentStateIdsForNN

    def __call__(self, state):
        stateNN = state[self.agentStateIdsForNN]
        stateBatch = [np.concatenate(stateNN)]
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionDist_ = graph.get_collection_ref("actionDistributions")[0]
        actionProbs = self.policyValueNet.run(actionDist_, feed_dict={state_: stateBatch})[0]
        actionDist = {action: prob for action, prob in zip(self.actionSpace, actionProbs)}
        return actionDist


class ApproximateValue:
    def __init__(self, policyValueNet,agentStateIdsForNN):
        self.policyValueNet = policyValueNet
        self.agentStateIdsForNN = agentStateIdsForNN

    def __call__(self, state):
        stateNN = state[self.agentStateIdsForNN]
        stateBatch = [np.concatenate(stateNN)]
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        valuePrediction_ = graph.get_collection_ref("values")[0]
        valuePrediction = self.policyValueNet.run(valuePrediction_, feed_dict={state_: stateBatch})[0][0]
        return valuePrediction


class ComposeMultiAgentTransitInSingleAgentMCTS:
    def __init__(self, chooseAction, reasonMindFunctions):
        self.chooseAction = chooseAction
        self.reasonMindFunctions = reasonMindFunctions

    def __call__(self,agentId, state, selfAction, othersPolicy, transit):
        reasonMind = list(self.reasonMindFunctions[agentId])
        del reasonMind[agentId]
        othersPolicyInMCTS = [reason(policy) for reason, policy in zip(reasonMind, othersPolicy )]
        multiAgentActions = [self.chooseAction(policy(state)) for policy in othersPolicyInMCTS]
        multiAgentActions.insert(agentId, selfAction)
        transitInSelfMCTS = transit(state, multiAgentActions)
        return transitInSelfMCTS

class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numTrees, numSimulationsPerTree, actionSpaces, agentIdsForNNState, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximateActionPrior, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS):
        self.numTrees = numTrees
        self.numSimulationsPerTree = numSimulationsPerTree
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximateActionPrior = getApproximateActionPrior
        self.getApproximateValue = getApproximateValue
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximateActionPrior(selfNNModel, self.actionSpaces[agentId], self.agentIdsForNNState[agentId])
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpaces[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue[agentId](selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = StochasticMCTS(self.numTrees, self.numSimulationsPerTree, self.selectChild, expand, estimateValue, backup, establishPlainActionDistFromMultipleTrees)

        return guidedMCTSPolicy

class ComposeSingleAgentMCTS():
    def __init__(self, numTrees, numSimulationsPerTree, actionSpaces, agentIdsForNNState, maxRolloutSteps, rewardFunctions, rolloutHeuristicFunctions, selectChild, isTerminal, transit, getApproximateActionPrior,composeMultiAgentTransitInSingleAgentMCTS):
        self.numTrees = numTrees
        self.numSimulationsPerTree = numSimulationsPerTree
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.maxRolloutSteps = maxRolloutSteps
        self.rewardFunctions = rewardFunctions
        self.rolloutHeuristicFunctions = rolloutHeuristicFunctions
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getApproximateActionPrior = getApproximateActionPrior
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximateActionPrior(selfNNModel, self.actionSpaces[agentId], self.agentIdsForNNState[agentId])
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpaces[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        rolloutPolicy = lambda state: self.actionSpaces[agentId][np.random.choice(range(len(self.actionSpaces[agentId])))]
        rewardFunction = self.rewardFunctions[agentId]
        rolloutHeuristic = self.rolloutHeuristicFunctions[agentId]
        estimateValue = RollOut(rolloutPolicy, self.maxRolloutSteps, transitInMCTS, rewardFunction, self.isTerminal, rolloutHeuristic)
        MCTSPolicy = StochasticMCTS(self.numTrees, self.numSimulationsPerTree, self.selectChild, expand, estimateValue, backup, establishPlainActionDistFromMultipleTrees)

        return MCTSPolicy

class PrepareMultiAgentPolicy:
    def __init__(self, MCTSAgentIds, actionSpaces, agentIdsForNNState, composeSingleAgentPolicy, getApproximatePolicy):
        self.MCTSAgentIds = MCTSAgentIds
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.composeSingleAgentPolicy = composeSingleAgentPolicy
        self.getApproximatePolicy = getApproximatePolicy

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy= np.array([self.getApproximatePolicy(NNModel, actionSpace, agentStateIdsForNN) for NNModel, actionSpace, agentStateIdsForNN in zip(multiAgentNNModel,
            self.actionSpaces, self.agentIdsForNNState)])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]])  for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentPolicy(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy) for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy



def main():
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data','generateExpDemo','trajectories')
    trajectoryPath = os.path.join(trajectoriesSaveDirectory, 'beta=0.5_demoIndex=0_killzoneRadius=0.5_masterPowerRatio=0.4_maxRunningSteps=360_numAgents=3_numSimulations=400_offset=0_pureMCTSAgentId=10.pickle')

    parametersForTrajectoryPath = {'masterPowerRatio':0.4, 'beta':0.5}

    masterPowerRatio = float(parametersForTrajectoryPath['masterPowerRatio'])
    beta = float(parametersForTrajectoryPath['beta'])

    # Mujoco environment
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'noRopeCollision3Agents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    draggerBodyIndex = 8

    #set physical model parameter
    physicsSimulation.model.body_mass[[draggerBodyIndex]] = [13]
    physicsSimulation.set_constants()
    physicsSimulation.forward()

    sheepId = 0
    wolfId = 1
    masterId = 2

    qPosIndex = [0, 1]
    getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
    getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)

    isTerminal = lambda state : False

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    transition = TransitConstantPhysics(transit)

    numAgent = 3
    numRopePart = 9
    ropePartIndex = list(range(numAgent, numAgent + numRopePart))

# neural network init
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numStateSpaceList = [24, 18, 18, 24]
    numActionSpace = len(actionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModels = [GenerateModel(numStateSpace, numActionSpace, regularizationFactor) for numStateSpace in numStateSpaceList]
    depth = 4

    initMultiAgentNNModels = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths) for generateModel in generateModels]

# sheep NN model
    sheepPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'sheepAvoidRopeModel','agentId=0_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
    sheepPreTrainModel = restoreVariables(initMultiAgentNNModels[sheepId], sheepPreTrainModelPath)

# wolf NN model
    wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
    wolfPreTrainModel = restoreVariables(initMultiAgentNNModels[wolfId], wolfPreTrainModelPath)

# master NN model
    masterPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedMasterNNModels','agentId=2_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
    masterPreTrainModel = restoreVariables(initMultiAgentNNModels[masterId], masterPreTrainModelPath)

    # multiAgentNNmodel = [sheepPreTrainModel, wolfPreTrainModel, masterPreTrainModel]


# MCTS compose
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

# multAgent ApproximatePolicyAndActionPrior
    preyPowerRatio = 1.15
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    predatorPowerRatio = 1.3
    wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    masterPowerRatio = masterPowerRatio
    masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))

    actionSpaceList = [sheepActionSpace, wolfActionSpace, masterActionSpace]
    # agentStateIdsForNNList = [range(4), range(3), range(3)]
    getApproximatePolicy = lambda NNmodel, actionSpace, agentStateIdsForNN: ApproximatePolicy(NNmodel, actionSpace, agentStateIdsForNN)
    getApproximateUniformActionPrior = lambda NNModel, actionSpace, agentStateIdsForNN: lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    aliveBonuses = [0.05, -0.05, 0.05]
    deathPenalties = [-1, 1, -1]

    agentIds = list(range(numAgent))
    getSelfQPoses = [GetAgentPosFromState(Id, qPosIndex) for Id in agentIds]
    otherIdsList = [[wolfId] + ropePartIndex, [sheepId], [sheepId]]#, [sheepId, wolfId, masterId] + ropePartIndex]
    getOthersPoses = [[GetAgentPosFromState(otherId, qPosIndex) for otherId in otherIds] for otherIds in otherIdsList]
    velIndex = [4, 5]
    getSelfVels =  [GetAgentPosFromState(Id, velIndex) for Id in agentIds]

    collisionRadius = 1
    isCollidedFunctions = [IsCollided(collisionRadius, getSelfQPos, getOthersPos) for getSelfQPos, getOthersPos in zip(getSelfQPoses, getOthersPoses)]
    safeBoundes = [2.5, 0.1, 0.1]#, 2]
    wallDisToCenter = 10
    wallPunishRatios = [3, 0, 0]#, 4]
    velocityBounds = [0, 0, 0]#, 6]
    rewardFunctions = [RewardFunctionAvoidCollisionAndWall(aliveBonuses[agentId], deathPenalties[agentId], safeBoundes[agentId], wallDisToCenter, \
            wallPunishRatios[agentId], velocityBounds[agentId], isCollidedFunctions[agentId], getSelfQPoses[agentId], getSelfVels[agentId]) \
            for agentId in agentIds]

    rolloutHeuristicWeights = [-0.1, 0.1, -0.1]#, 0]
    rolloutHeuristics = [HeuristicDistanceToTarget(weight, getWolfQPos, getSheepQPos) for weight in rolloutHeuristicWeights]

    numTrees = 2
    numSimulationsPerTree = 20
    maxRolloutSteps = 5

    betaInMCTS = 1
    chooseActionInMCTS = SampleAction(betaInMCTS)

    reasonMindList = np.array([[lambda policy: RandomPolicy(actionSpace) for actionSpace in actionSpaceList ] for subjectiveAgentId in range(numAgent)])

    reasonMindList[sheepId][wolfId] = lambda policy: policy
    reasonMindList[wolfId][sheepId] = lambda policy: policy

##################################################### try
    sheepStateIdsForNN = range(4)
    wolfStateIdsForNN = range(3)
    masterStateIdsForNN = range(3)

    getSheepApproximatePolicy = getApproximatePolicy(sheepPreTrainModel, sheepActionSpace, sheepStateIdsForNN)
    getWolfApproximatePolicy = getApproximatePolicy(wolfPreTrainModel, wolfActionSpace, wolfStateIdsForNN)
    getMasterApproximatePolicy = getApproximatePolicy(masterPreTrainModel, masterActionSpace, masterStateIdsForNN)

    multiAgentApproximatePolicy = np.array([getSheepApproximatePolicy, getWolfApproximatePolicy, getMasterApproximatePolicy])
    multiAgentNNmodel = [sheepPreTrainModel, wolfPreTrainModel, masterPreTrainModel]

    MCTSAgentIds = [sheepId, wolfId]
    otherAgentPolicyForMCTSAgents = np.array(
        [np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]]) for
         agentId in MCTSAgentIds])

    composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(chooseActionInMCTS, reasonMindList)
    composeSingleAgentMCTS = ComposeSingleAgentMCTS(numTrees, numSimulationsPerTree, actionSpaceList, agentStateIdsForNNList, maxRolloutSteps, rewardFunctions, rolloutHeuristics, selectChild, isTerminal, transit, getApproximateUniformActionPrior, composeMultiAgentTransitInSingleAgentMCTS)



    prepareMultiAgentPolicy = PrepareMultiAgentPolicy(trainableAgentIds, actionSpaceList, agentStateIdsForNNList, composeSingleAgentMCTS, getApproximatePolicy)


    policy = prepareMultiAgentPolicy(multiAgentNNmodel)

    # need function: policy(mind, state, allAgentsActions), output jointLikelihood (like 'ThreeAgentsPolicyForNN' in src/inferChasing/continuousPolicy)
    # mind = ['wolf', 'sheep', 'master']
    agentsNameList = ['wolf', 'sheep', 'master']
    policy = InferencePolicy(agentsNameList, agentsPolicyList, softenPolicy=None, softParam=None)

#####################################################

    getMindsPhysicsActionsJointLikelihood = lambda mind, state, allAgentsActions, physics, nextState: \
        policy(mind, state, allAgentsActions) * transition(physics, state, allAgentsActions, nextState)

    trajectories = loadFromPickle(trajectoryPath)
    dataIndex = 0
    trajectory = trajectories[dataIndex]

    chasingAgents = ['sheep', 'wolf', 'master']
    chasingSpace = list(it.permutations(chasingAgents))
    pullingSpace = ['constantPhysics']
    actionHypo = list(it.product(sheepActionSpace, wolfActionSpace, masterActionSpace))
    iterables = [chasingSpace, pullingSpace, actionHypo]
    inferenceIndex = pd.MultiIndex.from_product(iterables, names=['mind', 'physics', 'action'])

    thresholdPosterior = 1.5
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, inferenceIndex)

    decayParameter = 0.8
    mindPhysicsName = ['mind', 'physics']
    queryLikelihood = QueryDecayedLikelihood(mindPhysicsName, decayParameter)

    inferOneStepLikelihood = InferOneStepLikelihood(inferenceIndex, getMindsPhysicsActionsJointLikelihood)

    stateIndex = 0
    observe = Observe(stateIndex, trajectory)
    FPS = 60
    inferContinuousChasingAndDrawDemo = InferContinuousChasingAndDrawDemo(FPS, inferenceIndex,
                                                                          isInferenceTerminal, observe, queryLikelihood,
                                                                          inferOneStepLikelihood)
    numOfAgents = 3
    mindsPhysicsPrior = [1 / len(inferenceIndex)] * len(inferenceIndex)
    posteriorDf = inferContinuousChasingAndDrawDemo(numOfAgents, mindsPhysicsPrior)

    plotMindInferenceProb = PlotInferenceProb('timeStep', 'mindPosterior', 'mind')
    plotPhysicsInferenceProb = PlotInferenceProb('timeStep', 'physicsPosterior', 'physics')


    plotName = 'NoRopeCollision3AgentsInference'
    plotMindInferenceProb(posteriorDf, dataIndex, plotName)
    plotPhysicsInferenceProb(posteriorDf, dataIndex, plotName)



if __name__ == '__main__':
    main()