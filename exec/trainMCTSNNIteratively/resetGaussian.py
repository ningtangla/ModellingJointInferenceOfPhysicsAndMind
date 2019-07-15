import numpy as np

class ResetGaussian:
    def __init__(self, simulation, qVelInit, numAgent, qPosInitstdDev, qVelInitNoise, qMin, qMax, withinBounds):
        self.simulation = simulation
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = numAgent
        self.qPosInitstdDev = qPosInitstdDev
        self.qVelInitNoise = qVelInitNoise
        self.qMin = qMin
        self.qMax = qMax
        self.withinBounds = withinBounds

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)
        numQPosEachAgent = int(numQPos / self.numAgent)
        numQVelEachAgent = int(numQVel / self.numAgent)

        while True:
            agentQPosInit = np.random.uniform(self.qMin, self.qMax, numQPosEachAgent)
            qPosInit = np.tile(agentQPosInit, self.numAgent)
            qPos = np.random.normal(loc=qPosInit, scale=self.qPosInitstdDev, size=numQPos)
            qVel = qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)
            if(self.withinBounds(qPos)):
                break

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        xPos = np.concatenate(self.simulation.data.body_xpos[-self.numAgent:, :numQPosEachAgent])

        agentQPos = lambda agentIndex: qPos[numQPosEachAgent * agentIndex: numQPosEachAgent * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[numQPosEachAgent * agentIndex: numQPosEachAgent * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[numQVelEachAgent * agentIndex: numQVelEachAgent * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate(
            [agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState


