import numpy as np


class RemoveNoiseFromState:
    def __init__(self, noiseIndex):
        self.noiseIndex = noiseIndex

    def __call__(self, trajectoryState):
        state = [np.delete(state, self.noiseIndex) for state in trajectoryState]
        return np.asarray(state).flatten()


class ProcessTrajectoryForNN:
    def __init__(self, agentId, actionToOneHot, removeNoiseFromState):
        self.agentId = agentId
        self.actionToOneHot = actionToOneHot
        self.removeNoiseFromState = removeNoiseFromState

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist, value: \
            (self.removeNoiseFromState(state), self.actionToOneHot(actions[self.agentId]), value)
        processedTrajectory = [processTuple(*triple) for triple in trajectory]

        return processedTrajectory


class PreProcessTrajectories:
    def __init__(self, addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN):
        self.addValuesToTrajectory = addValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory
        self.processTrajectoryForNN = processTrajectoryForNN

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addValuesToTrajectory(trajectory) for trajectory in trajectories]
        filteredTrajectories = [self.removeTerminalTupleFromTrajectory(trajectory) for trajectory in trajectoriesWithValues]
        processedTrajectories = [self.processTrajectoryForNN(trajectory) for trajectory in filteredTrajectories]
        allDataPoints = [dataPoint for trajectory in processedTrajectories for dataPoint in trajectory]
        trainData = [list(varBatch) for varBatch in zip(*allDataPoints)]

        return trainData