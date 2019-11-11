import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
import numpy as np
from ddt import ddt, data, unpack
from mujoco_py import load_model_from_path, MjSim

# Local import
from src.constrainedChasingEscapingEnv.envMujoco import Reset, TransitionFunction, IsTerminal, WithinBounds
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState


class checkAngentStackInWall:
    def __init__(self, wallList,agentMaxSize):
        self.wallList=wallList
        self.agentMaxSize=agentMaxSize
    def call (qPosList):
        wallCenterList=np.array([wall[:2] for wall in wall List])
        wallExpandHalfDiagonalList=np.array([np.add(wall[2:],self.agentMaxSize)] for wall in wallList])
        posList=qPosList.reshape(-1,2)
        isOverlapList=[np.all(np.abs(np.add(pos,-center))<diag)  for (center,diag) in zip (wallCenterList,wallExpandHalfDiagonalList) for pos in posList]
        return=np.any(posList)

class ResetUniformInEnvWithObstacles:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise,checkAngentStackInWall):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)

    def __call__(self):
        qPos = self.qPosInit + np.concatenate([np.random.uniform(low=-np.array(self.qPosInitNoise), high=np.array(self.qPosInitNoise)) for agentId in range(self.numAgent)])
        while self.checkAngentStackInWall(qPos):
            qPos = self.qPosInit + np.concatenate([np.random.uniform(low=-np.array(self.qPosInitNoise), high=np.array(self.qPosInitNoise)) for agentId in range(self.numAgent)])
        qVel = self.qVelInit + np.concatenate([np.random.uniform(low=-np.array(self.qVelInitNoise), high=np.array(self.qVelInitNoise)) for agentId in range(self.numAgent)])
        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        xPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgent, :self.numJointEachSite])

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate(
            [agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState


@ddt
class TestEnvMujoco(unittest.TestCase):
    def setUp(self):
        self.modelPath = os.path.join(DIRNAME, '..', 'env', 'xmls', 'twoAgents.xml')
        self.model = load_model_from_path(self.modelPath)
        self.simulation = MjSim(self.model)
        self.numJointEachAgent = 2
        self.numAgent = 2
        self.killzoneRadius = 0.5
        self.numSimulationFrames = 20
        self.sheepId = 0
        self.wolfId = 1
        self.xPosIndex = [2, 3]
        self.getSheepPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)
        self.isTerminal = IsTerminal(self.killzoneRadius, self.getSheepPos, self.getWolfPos)
        self.minQPos = (-9.7, -9.7)
        self.maxQPos = (9.7, 9.7)
        self.withinBounds = WithinBounds(self.minQPos, self.maxQPos)

    @data(([0, 0, 0, 0], [0, 0, 0, 0], np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])),
          ([1, 2, 3, 4], [0, 0, 0, 0], np.asarray([[1, 2, 1, 2, 0, 0], [3, 4, 3, 4, 0, 0]])),
          ([1, 2, 3, 4], [5, 6, 7, 8], np.asarray([[1, 2, 1, 2, 5, 6], [3, 4, 3, 4, 7, 8]])))
    @unpack
    def testResetUniform(self, qPosInit, qVelInit, groundTruthReturnedInitialState):
        resetUniform = ResetUniform(self.simulation, qPosInit, qVelInit, self.numAgent)
        returnedInitialState = resetUniform()
        truthValue = returnedInitialState == groundTruthReturnedInitialState
        self.assertTrue(truthValue.all())

