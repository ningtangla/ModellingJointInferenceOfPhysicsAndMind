import mujoco_py as mujoco
import os
import numpy as np

class Reset():
    def __init__(self, modelName, qPosInit, qVelInit, numAgent, qPosInitNoise=0, qVelInitNoise=0):
        dirName = os.path.dirname(__file__)
        model = mujoco.load_model_from_path(os.path.join(dirName, '..', '..', 'env', 'xmls', '{}.xml'.format(modelName)))
        self.simulation = mujoco.MjSim(model)
        self.qPosInit = qPosInit
        self.qVelInit = qVelInit
        self.numAgent = numAgent
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)
        numQPosEachAgent = int(numQPos/self.numAgent)
        numQVelEachAgent = int(numQVel/self.numAgent)

        qPos = np.array(self.qPosInit) + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = np.array(self.qVelInit) + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()
        xPos = np.concatenate(self.simulation.data.body_xpos[-self.numAgent: , :numQPosEachAgent])
        startState = np.array([np.concatenate([qPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)], xPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)],
            qVel[numQVelEachAgent * agentIndex : numQVelEachAgent * (agentIndex + 1)]]) for agentIndex in range(self.numAgent)])

        return startState


class TransitionFunction:
    def __init__(self, modelName, isTerminal, renderOn, numSimulationFrames):
        dirName = os.path.dirname(__file__)
        model = mujoco.load_model_from_path(os.path.join(dirName, '..', '..', 'env', 'xmls', '{}.xml'.format(modelName)))
        self.simulation = mujoco.MjSim(model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)
        self.renderOn = renderOn
        if self.renderOn:
            self.viewer = mujoco.MjViewer(self.simulation)
            self.frames = []

        self.isTerminal = isTerminal
        self.numSimulationFrames = numSimulationFrames
        
    def __call__(self, worldState, allAgentsActions):
        numAgent = len(worldState)
        numQPosEachAgent = int(self.numQPos/numAgent)
        numQVelEachAgent = int(self.numQVel/numAgent)

        allAgentOldQPos = worldState[:, 0:numQPosEachAgent].flatten()
        allAgentOldQVel = worldState[:, -numQVelEachAgent:].flatten()

        self.simulation.data.qpos[:] = allAgentOldQPos
        self.simulation.data.qvel[:] = allAgentOldQVel
        self.simulation.data.ctrl[:] = np.asarray(allAgentsActions).flatten()

        for i in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()
            if self.renderOn:
                frame = self.simulation.render(1024, 1024, camera_name="center")
                self.frames.append(frame)

            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
            newXPos = np.concatenate(self.simulation.data.body_xpos[-numAgent: , :numQPosEachAgent])

            newState = np.array([np.concatenate([newQPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)], newXPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)],
                        newQVel[numQVelEachAgent * agentIndex : numQVelEachAgent * (agentIndex + 1)]]) for agentIndex in range(numAgent)])

            if self.isTerminal(newState):
                break

        return newState


def euclideanDistance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))


class IsTerminal():
    def __init__(self, minXDis, getAgent0Pos, getAgent1Pos):
        self.minXDis = minXDis
        self.getAgent0Pos = getAgent0Pos
        self.getAgent1Pos = getAgent1Pos

    def __call__(self, state):
        pos0 = self.getAgent0Pos(state)
        pos1 = self.getAgent1Pos(state)
        distance = euclideanDistance(pos0, pos1)
        terminal = (distance <= self.minXDis)

        return terminal