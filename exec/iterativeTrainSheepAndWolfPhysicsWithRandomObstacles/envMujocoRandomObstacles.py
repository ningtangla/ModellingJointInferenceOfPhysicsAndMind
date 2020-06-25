import numpy as np
import math


class SampleObscalesProperty():
    def __init__(self, sampleWallPos,sampleWallSize):
        self.sampleWallPos = sampleWallPos
        self.sampleWallSize = sampleWallSize
    def __call__(self):
        wallPosList=self.sampleWallPos()
        wallSizeList=self.sampleWallSize()
        return wallPosList, wallSizeList

def changeWallProperty(envDict,wallPropertyDict):
    for number,propertyDict in wallPropertyDict.items():
        for name,value in propertyDict.items():
            envDict['mujoco']['worldbody']['body'][number]['geom'][name]=value

    return envDict

class SetMujocoEnvXmlProperty():
    def __init__(self, wallIdlist,changeWallProperty):
        def transferNumberListToStr(numList):
            strList=[str(num) for num in numList]
            return ' '.join(strList)
        self.wallIdlist=wallIdlist
        self.transferNumberListToStr=transferNumberListToStr
        self.changeWallProperty = changeWallProperty

    def __call__(self,wallPosList,wallSizeList,xml_doc_dict):
        wallPropertyDict={}
        [wallPropertyDict.update({wallId:{'@pos':self.transferNumberListToStr(wallPos),'@size':self.transferNumberListToStr(wallSize)}}) for (wallId,wallPos,wallSize) in zip(self.wallIdlist,wallPosList,wallSizeList)]
        xml_doc_dict=changeWallProperty(xml_doc_dict,wallPropertyDict)
        return xml_doc_dict


def getWallList(wallPosList,wallSizeList):
    return [pos[:2]+size[:2] for (pos,size) in zip(wallPosList,wallSizeList)]

class CheckAngentStackInWall:
    def __init__(self,agentMaxSize):

        self.agentMaxSize=agentMaxSize
    def __call__(self,qPosList,wallList):
        wallCenterList=np.array([wall[:2] for wall in wallList])
        wallExpandHalfDiagonalList=np.array([np.add(wall[2:],self.agentMaxSize) for wall in wallList])
        posList=qPosList.reshape(-1,2)
        isOverlapList=[np.all(np.abs(np.add(pos,-center))<diag)  for (center,diag) in zip (wallCenterList,wallExpandHalfDiagonalList) for pos in posList]
        return np.any(isOverlapList)

class ResetUniformInEnvWithObstacles:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise,obstacleProperty,checkAngentStackInWall):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.obstacleProperty=[np.asarray(obstacle) for obstacle in obstacleProperty]
        self.checkAngentStackInWall=checkAngentStackInWall
    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)


        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)

        while self.checkAngentStackInWall(qPos,self.obstacleProperty):
            qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        xPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgent, :self.numJointEachSite])

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        getAgentState = lambda agentIndex: np.concatenate(
            [agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])

        # startState = np.asarray([getAgentState(agentIndex) for agentIndex in range(self.numAgent)])

        agentState = [getAgentState(agentIndex) for agentIndex in range(self.numAgent)]

        startState=np.asarray(agentState+self.obstacleProperty)
        return startState


class SampleRandomWallSize():
    def __init__(self, gapDelta,wallLengthDelta,wallWidthDelta,zPos=-0.2,zSize=1.5):
        self.gapDelta = gapDelta
        self.wallLengthDelta=wallLengthDelta
        self.wallWidthDelta=wallWidthDelta
        self.zPos=zPos
        self.zSize=zSize

    def __call__(self):

        halfGapLen=np.random.uniform(self.gapDelta[0],self.gapDelta[1],size=2)
        halfWallLength=np.random.uniform(self.wallLengthDelta[0],self.wallLengthDelta[1],size=2)
        halfWallWidth=np.random.uniform(self.wallWidthDelta[0],self.wallWidthDelta[1],size=2)

        initWallPosList=[[0,dirc*(gap+lenth),self.zPos] for gap,lenth, dirc in zip(halfGapLen,halfWallLength,[-1,1])]
        initWallSizeList=[[width,lenth,self.zSize] for width,lenth in zip (halfWallWidth,halfWallLength)]



        return initWallPosList,initWallSizeList


class RandomMoveObstacleCenter(object):
    def __init__(self, initWallPosList,wallXdelta,wallYdelta):
        self.initWallPosList = initWallPosList
        self.wallXdelta = wallXdelta
        self.wallYdelta = wallYdelta
    def __call__(self):
        x=np.random.uniform(self.wallXdelta[0],self.wallXdelta[1],size=1)[0]
        y=np.random.uniform(self.wallYdelta[0],self.wallYdelta[1],size=1)[0]
        movingVector=[x,y,0]
        wallPosList=[[pos+delta for pos,delta in zip(obstacle,movingVector)] for obstacle in self.initWallPosList]
        return wallPosList

class CheckObstacleOutEnv():
    def __init__(self, initWallSizeList,allowedArea):
        self.initWallSizeList = initWallSizeList
        self.allowedArea = allowedArea
    def __call__(self,):




        return np.any(isOverlapList)

class CheckObstacleOutEnv():
    def __init__(self, initWallSizeList,allowedArea):
        self.initWallSizeList = initWallSizeList
        self.allowedArea = allowedArea
        centerXBoudaries=[[bound-wallsize[0]*dirc for bound,dirc in zip(allowedArea[0],[-1,1] )]for wallsize  in initWallSizeList] 
        self.centerXBoudary=[max(centerXBoudaries[0][0],centerXBoudaries[1][0] ),min(centerXBoudaries[0][1],centerXBoudaries[1][1] )]

        self.centerYBoudary=[bound-wallsize[1]*dirc for wallsize,bound ,dirc in zip(initWallSizeList,allowedArea[1],[-1,1] )]
        
    def __call__(self, wallPosList):
        for wallPos in wallPosList:
            if wallPos[0]<self.centerXBoudary[0] or wallPos[0]>self.centerXBoudary[1] :
                print('xout',wallPos[0],self.centerXBoudary)
                return False
            if wallPos[1]<self.centerYBoudary[0] or wallPos[1]>self.centerYBoudary[1] :
                print('yout',wallPos[0],self.centerYBoudary)
                return False

        return True



class RejectSampleObstacleMoveVector(object):
    def __init__(self, sampleMoveVector,checkObstacleInEnv):
        self.sampleMoveVector = sampleMoveVector
        self.checkObstacleInEnv = checkObstacleInEnv
        
    def __call__(self):

        wallPosList=self.sampleMoveVector()
        while not self.checkObstacleInEnv(wallPosList):
            wallPosList=self.sampleMoveVector()
        return wallPosList

class ResetUniformInEnvWithRandomObstacles:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise,obstacleProperty,checkAngentStackInWall):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.obstacleProperty=[np.asarray(obstacle) for obstacle in obstacleProperty]
        self.checkAngentStackInWall=checkAngentStackInWall
    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)


        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)

        while self.checkAngentStackInWall(qPos,self.obstacleProperty):
            qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        xPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgent, :self.numJointEachSite])

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        getAgentState = lambda agentIndex: np.concatenate(
            [agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])

        # startState = np.asarray([getAgentState(agentIndex) for agentIndex in range(self.numAgent)])

        agentState = [getAgentState(agentIndex) for agentIndex in range(self.numAgent)]

        startState=np.asarray(agentState+self.obstacleProperty)
        return startState
class TransitionFunction:
    def __init__(self,numAgents,simulation, numSimulationFrames,isTerminal):
        self.numAgents=numAgents
        self.simulation = simulation
        self.numSimulationFrames = numSimulationFrames
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)
        self.isTerminal = isTerminal
    def __call__(self, state, actions):
        state = np.asarray(state)
        # print("state", state)
        actions = np.asarray(actions)
        numAgent = len(state)

        # oldQPos = state[:, 0:self.numJointEachSite].flatten()
        # oldQVel = state[:, -self.numJointEachSite:].flatten()
        # oldQPos = state[:self.numAgents, 0:self.numJointEachSite].flatten()
        # oldQVel = state[:self.numAgents, self.numJointEachSite:2*self.numJointEachSite].flatten()
        oldQPos =np.array([QPos for agent in state[:self.numAgents] for QPos in agent[:self.numJointEachSite]]).flatten()
        oldQVel =np.array([QVel for agent in state[:self.numAgents] for QVel in agent[-self.numJointEachSite:]]).flatten()
        obstacleProperty=[np.asarray(obstacle) for obstacle in state[self.numAgents:]]

        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = actions.flatten()

        for simulationFrame in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()

            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
            newXPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgents, :self.numJointEachSite])

            agentNewQPos = lambda agentIndex: newQPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (
                        agentIndex + 1)]
            agentNewXPos = lambda agentIndex: newXPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (
                        agentIndex + 1)]
            agentNewQVel = lambda agentIndex: newQVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (
                        agentIndex + 1)]
            getAgentNewState = lambda agentIndex: np.concatenate([agentNewQPos(agentIndex), agentNewXPos(agentIndex),agentNewQVel(agentIndex)])

            agentNewState=[getAgentNewState(agentIndex) for agentIndex in range(self.numAgents)]
            # print('obs',obstacleProperty)
            # newState = np.asarray([agentNewState(agentIndex) for agentIndex in range(numAgent)])

            # newState = np.concatenate(agentNewState,obstacleProperty)
            newState=np.asarray(agentNewState+obstacleProperty)
            # print('new',newState)
            if self.isTerminal(newState):
                break

        return newState