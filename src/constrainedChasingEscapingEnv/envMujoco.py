import numpy as np
import math


class ResetUniform:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise=0, qVelInitNoise=0):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)

        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        xPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgent, :self.numJointEachSite])

        agentQPos = lambda agentIndex: qPos[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate(
            [agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState


class ResetUniformForLeashed:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, tiedAgentIndex, ropePartIndex, maxRopePartLength,
                 qPosInitNoise=0, qVelInitNoise=0):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.tiedBasePosAgentIndex, self.tiedFollowPosAgentIndex = tiedAgentIndex
        self.numRopePart = len(ropePartIndex)
        self.maxRopePartLength = maxRopePartLength
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)

        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        tiedBasePos = qPos[self.numJointEachSite * self.tiedBasePosAgentIndex: self.numJointEachSite * (self.tiedBasePosAgentIndex + 1)]
        sampledRopeLength = np.random.uniform(low = 0, high = self.numRopePart * self.maxRopePartLength)
        sampledPartLength = np.arange(sampledRopeLength/(self.numRopePart + 1), sampledRopeLength, sampledRopeLength/(self.numRopePart + 1))[:self.numRopePart]
        theta = np.random.uniform(low = 0, high = math.pi)

        tiedFollowPosAgentPos = tiedBasePos + np.array([sampledRopeLength * np.cos(theta), sampledRopeLength * np.sin(theta)])
        qPos[self.numJointEachSite * self.tiedFollowPosAgentIndex : self.numJointEachSite * (self.tiedFollowPosAgentIndex + 1)] = tiedFollowPosAgentPos
        ropePartPos = np.array(list(zip(sampledPartLength * np.cos(theta), sampledPartLength * np.sin(theta)))) + tiedBasePos
        qPos[-self.numJointEachSite * self.numRopePart : ] = np.concatenate(ropePartPos)

        qVelSampled = np.concatenate([np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel - self.numRopePart * self.numJointEachSite),\
                                      np.zeros(self.numRopePart * self.numJointEachSite)])
        qVel = self.qVelInit + qVelSampled

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        xPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgent, :self.numJointEachSite])

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState


class TransitionFunction:
    def __init__(self, simulation, isTerminal, numSimulationFrames):
        self.simulation = simulation
        self.isTerminal = isTerminal
        self.numSimulationFrames = numSimulationFrames
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)

    def __call__(self, state, actions):
        state = np.asarray(state)
        # print("state", state)
        actions = np.asarray(actions)
        numAgent = len(state)

        oldQPos = state[:, 0:self.numJointEachSite].flatten()
        oldQVel = state[:, -self.numJointEachSite:].flatten()

        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = actions.flatten()

        for simulationFrame in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()

            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
            newXPos = np.concatenate(self.simulation.data.site_xpos[:numAgent, :self.numJointEachSite])

            agentNewQPos = lambda agentIndex: newQPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (
                        agentIndex + 1)]
            agentNewXPos = lambda agentIndex: newXPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (
                        agentIndex + 1)]
            agentNewQVel = lambda agentIndex: newQVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (
                        agentIndex + 1)]
            agentNewState = lambda agentIndex: np.concatenate([agentNewQPos(agentIndex), agentNewXPos(agentIndex),
                                                               agentNewQVel(agentIndex)])
            newState = np.asarray([agentNewState(agentIndex) for agentIndex in range(numAgent)])

            if self.isTerminal(newState):
                break

        return newState


class ResetUniformWithoutXPos:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise=0, qVelInitNoise=0):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)

        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        agentQPos = lambda agentIndex: qPos[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState


class ResetUniformWithoutXPosForLeashed:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, tiedAgentIndex, ropePartIndex, maxRopePartLength,
                 qPosInitNoise=0, qVelInitNoise=0):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.tiedBasePosAgentIndex, self.tiedFollowPosAgentIndex = tiedAgentIndex
        self.numRopePart = len(ropePartIndex)
        self.maxRopePartLength = maxRopePartLength
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)

        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        tiedBasePos = qPos[self.numJointEachSite * self.tiedBasePosAgentIndex: self.numJointEachSite * (self.tiedBasePosAgentIndex + 1)]
        sampledRopeLength = np.random.uniform(low = 0, high = self.numRopePart * self.maxRopePartLength)
        sampledPartLength = np.arange(sampledRopeLength/(self.numRopePart + 1), sampledRopeLength, sampledRopeLength/(self.numRopePart + 1))[:self.numRopePart]
        theta = np.random.uniform(low = 0, high = math.pi)

        tiedFollowPosAgentPos = tiedBasePos + np.array([sampledRopeLength * np.cos(theta), sampledRopeLength * np.sin(theta)])
        qPos[self.numJointEachSite * self.tiedFollowPosAgentIndex : self.numJointEachSite * (self.tiedFollowPosAgentIndex + 1)] = tiedFollowPosAgentPos
        ropePartPos = np.array(list(zip(sampledPartLength * np.cos(theta), sampledPartLength * np.sin(theta)))) + tiedBasePos
        qPos[-self.numJointEachSite * self.numRopePart : ] = np.concatenate(ropePartPos)

        qVelSampled = np.concatenate([np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel - self.numRopePart * self.numJointEachSite),\
                                      np.zeros(self.numRopePart * self.numJointEachSite)])
        qVel = self.qVelInit + qVelSampled

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        agentQPos = lambda agentIndex: qPos[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[
                                       self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState


class TransitionFunctionWithoutXPos:
    def __init__(self, simulation, isTerminal, numSimulationFrames):
        self.simulation = simulation
        self.isTerminal = isTerminal
        self.numSimulationFrames = numSimulationFrames
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)

    def __call__(self, state, actions):
        state = np.asarray(state)
        # print("state", state)
        actions = np.asarray(actions)
        numAgent = len(state)

        oldQPos = state[:, 0:self.numJointEachSite].flatten()
        oldQVel = state[:, -self.numJointEachSite:].flatten()

        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = actions.flatten()
        for simulationFrame in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()

            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel

            agentNewQPos = lambda agentIndex: newQPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (
                        agentIndex + 1)]
            agentNewQVel = lambda agentIndex: newQVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (
                        agentIndex + 1)]
            agentNewState = lambda agentIndex: np.concatenate([agentNewQPos(agentIndex), agentNewQVel(agentIndex)])
            newState = np.asarray([agentNewState(agentIndex) for agentIndex in range(numAgent)])

            if self.isTerminal(newState):
                break

        return newState


class Transition3Objects:
    def __init__(self, simulation, numSimulationFrames):
        self.simulation = simulation
        self.numSimulationFrames = numSimulationFrames
        self.numJointEachSite = self.simulation.model.njnt / self.simulation.model.nsite

    def __call__(self, state, actions):
        state = np.asarray(state)
        # print("state", state)
        actions = np.asarray(actions)
        numAgent = len(state)

        oldQPos = state[:, 0:self.numJointEachSite].flatten()
        oldQVel = state[:, -self.numJointEachSite:].flatten()

        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = actions.flatten()

        for simulationFrame in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()

            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
            newXPos = np.concatenate(self.simulation.data.body_xpos[-numAgent:, :self.numJointEachSite])

            agentNewQPos = lambda agentIndex: newQPos[
                                              self.numJointEachSite * agentIndex: self.numJointEachSite * (
                                                          agentIndex + 1)]
            agentNewXPos = lambda agentIndex: newXPos[
                                              self.numJointEachSite * agentIndex: self.numJointEachSite * (
                                                          agentIndex + 1)]
            agentNewQVel = lambda agentIndex: newQVel[
                                              self.numJointEachSite * agentIndex: self.numJointEachSite * (
                                                          agentIndex + 1)]
            agentNewState = lambda agentIndex: np.concatenate([agentNewQPos(agentIndex), agentNewXPos(agentIndex),
                                                               agentNewQVel(agentIndex)])
            newState = np.asarray([agentNewState(agentIndex) for agentIndex in range(numAgent)])

        return newState


class IsTerminal:
    def __init__(self, minXDis, getAgent0Pos, getAgent1Pos):
        self.minXDis = minXDis
        self.getAgent0Pos = getAgent0Pos
        self.getAgent1Pos = getAgent1Pos

    def __call__(self, state):
        state = np.asarray(state)
        pos0 = self.getAgent0Pos(state)
        pos1 = self.getAgent1Pos(state)
        L2Normdistance = np.linalg.norm((pos0 - pos1), ord=2)
        terminal = (L2Normdistance <= self.minXDis)

        return terminal


class WithinBounds:
    def __init__(self, minQPos, maxQPos):
        self.minQPos = np.asarray(minQPos)
        self.maxQPos = np.asarray(maxQPos)

    def __call__(self, qPos):
        qPos = np.asarray(qPos)
        self.numJointEachSite = len(self.minQPos)
        numQPos = len(qPos)
        numAgents = int(numQPos / self.numJointEachSite)
        getAgentQPos = lambda agentIndex: qPos[
                                          self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentWithinBounds = lambda agentIndex: np.all(np.less_equal(getAgentQPos(agentIndex), self.maxQPos)) and \
                                               np.all(np.greater_equal(getAgentQPos(agentIndex), self.minQPos))
        allAgentsWithinbounds = all(agentWithinBounds(agentIndex) for agentIndex in range(numAgents))

        return allAgentsWithinbounds
