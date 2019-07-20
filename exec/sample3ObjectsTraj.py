import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from src.constrainedChasingEscapingEnv.envMujoco import Transition3Objects
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximateActionPrior
from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform
from src.episode import Sample3ObjectsTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import saveToPickle
from src.inferChasing.continuousPolicy import RandomPolicy


import pandas as pd
import mujoco_py as mujoco
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main():
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'threeAgents.xml')
    agentsBodyMassIndex = [6, 7, 8]
    physicsSmallMassModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSmallMassModel.body_mass[agentsBodyMassIndex] = [4, 5, 4]
    physicsLargeMassModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsLargeMassModel.body_mass[agentsBodyMassIndex] = [8, 10, 8]

    physicsSmallMassSimulation = mujoco.MjSim(physicsSmallMassModel)
    physicsLargeMassSimulation = mujoco.MjSim(physicsLargeMassModel)
    # set_constants fit for mujoco_py version >= 2.0, no fit for 1.50
    physicsSmallMassSimulation.set_constants()
    physicsLargeMassSimulation.set_constants()

    numSimulationFrames = 20
    transitSmallMassAgents = Transition3Objects(physicsSmallMassSimulation, numSimulationFrames)
    transitLargeMassAgents = Transition3Objects(physicsLargeMassSimulation, numSimulationFrames)

    transit = transitSmallMassAgents


    qPosInit = (0, 0, 0, 0, 0, 0)     # (initial position of sheep, initial position of wolf)
    qVelInit = (0, 0, 0, 0, 0, 0)     # (initial velocity of sheep, initial velocity of wolf)
    qPosInitNoise = 9.7         # adds some randomness to the initial positions
    qVelInitNoise = 5           # adds some randomness to the initial velocities
    numAgent = 3
    reset = ResetUniform(physicsSmallMassSimulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise)

    # sample trajectory
    maxRunningSteps = 20        # max possible length of the trajectory/episode
    sampleTrajectory = Sample3ObjectsTrajectory(maxRunningSteps, transit, reset, chooseGreedyAction)

    # Neural Network
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # wolf NN Policy
    wolfModelPath = os.path.join(dirName, '..','NNModels','wolfNNModels', 'killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999')
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(wolfNNModel, wolfModelPath)
    wolfPolicy = ApproximateActionPrior(wolfNNModel, actionSpace) # input state, return action distribution

    # sheep NN Policy
    sheepModelPath = os.path.join(dirName, '..','NNModels','sheepNNModels', 'killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999')
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    sheepPolicy = ApproximateActionPrior(sheepNNModel, actionSpace) # input sheepstate, return action distribution

    randomPolicy = RandomPolicy(actionSpace)

    policy = lambda state: [wolfPolicy(state[:2]), sheepPolicy(state[:2]), randomPolicy(state)]

    trajectory = sampleTrajectory(policy)
    dataIndex = 15
    dataPath = os.path.join(dirName, '..', 'trainedData', 'trajectory'+ str(dataIndex) + '.pickle')
    saveToPickle(trajectory, dataPath)





if __name__ == '__main__':
    main()





































