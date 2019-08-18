
import numpy as np
import pandas as pd
import os
import math

def calculateIncludedAngle(vector1, vector2):
    includedAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
    return includedAngle

class CalculateChasingDeviation:
    def __init__(self, sheepId, wolfId, stateIndex, positionIndex):
        self.sheepId = sheepId
        self.wolfId = wolfId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex

    def __call__(self, traj):
        traj = np.array(traj)
        heatSeekingVectorList = [traj[i - 1][self.stateIndex][self.sheepId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]
        wolfVectorList = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]

        sheepWolfAngleList = np.array([calculateIncludedAngle(v1, v2) for (v1, v2) in zip(heatSeekingVectorList, wolfVectorList)])
        return np.mean(sheepWolfAngleList)

class IsAllInAngelRange():
    def __init__(self, lowBound, upBound):
        self.lowBound = lowBound
        self.upBound = upBound
    def __call__(self, angleList):
        filterangleList=list(filter(lambda x: x<=self.upBound and x>=self.lowBound,angleList))
        return len(filterangleList)<len(angleList)

class CountCirclesBetweenWolfAndMaster:
    def __init__(self, wolfId, masterId,stateIndex, positionIndex,timeWindow,findCirleMove):
        self.wolfId = wolfId
        self.masterId=masterId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex
        self.timeWindow=timeWindow
        self.findCirleMove=findCirleMove

    def __call__(self, traj):
        traj = np.array(traj)
        wolfVectorList = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]
        ropeVectorList = [traj[i-1][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.masterId][self.positionIndex] for i in range(1, len(traj)) ]
        ropeWolfAngleList = np.array([calculateIncludedAngle(v1, v2) for (v1, v2) in zip(wolfVectorList, ropeVectorList)])
        filterList=[self.findCirleMove(ropeWolfAngleList[i:i+self.timeWindow]) for i in range(0, len(ropeWolfAngleList)-self.timeWindow+1) ]
        # countList=[ filterList[i]!=filterList[i+1] for i in range(0, len(filterList)-1) ]
        countList = filterList
        ciclrNumber=math.ceil(len(np.where(countList)[0])/2)
        ciclrRatio = ciclrNumber/len(traj)
        return ciclrRatio

class CountSheepCrossRope:
    def __init__(self,sheepId, wolfId, masterId,stateIndex, positionIndex,tranformCoordinates,isCrossAxis):
        self.sheepId =sheepId
        self.wolfId = wolfId
        self.masterId=masterId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex
        self.tranformCoordinates=tranformCoordinates
        self.isCrossAxis=isCrossAxis
    def __call__(self, traj):
        ropeVectorList = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] - traj[i][self.stateIndex][self.masterId][self.positionIndex] for i in range(len(traj)-1)]
        sheepMasterVectorList = [traj[i][self.stateIndex][self.sheepId][self.positionIndex] - traj[i][self.stateIndex][self.masterId][self.positionIndex] for i in range(len(traj)-1)]
        tanformeredSheepCoordatesList =[self.tranformCoordinates(xaxis, sheepvector) for (xaxis, sheepvector) in zip(ropeVectorList, sheepMasterVectorList)]
        crossList=[self.isCrossAxis(tanformeredSheepCoordatesList[i],tanformeredSheepCoordatesList[i+1]) for i in range(len(tanformeredSheepCoordatesList)-1)]
        crossNumber=len(np.where(crossList)[0])
        crossRatio = crossNumber/ len(traj)
        return crossRatio
        
def tranformCoordinates(axisVerctor,toTransformVecor):
    newCoorComplex=complex(toTransformVecor[0], toTransformVecor[1]) / complex(axisVerctor[0], axisVerctor[1])
    return [newCoorComplex.real,newCoorComplex.imag]

def isCrossAxis(v1,v2):
    x1=v1[0]
    y1=v1[1]
    x2=v2[0]
    y2=v2[1]
    if y1==y2 :
        if y1==0:
            lower=min(x1,x2)
            uper=max(x1,x2)
            return ((lower<=1) and (uper>=0))
        else:
            return True
    crossX=-y1*(x2-x1)/(y2-y1)+x1
    return  (crossX<1 and crossX>0 and y1*y2<0)

class CountSheepInCorner :
    def __init__(self,sheepId,stateIndex, positionIndex,spaceSize,cornerSize,isInCorner):
        self.sheepId =sheepId  
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex
        self.spaceSize=spaceSize
        self.cornerSize=cornerSize
        self.isInCorner=isInCorner
    def __call__(self,traj):

        sheepCoordinateList = [traj[i][self.stateIndex][self.sheepId][self.positionIndex] for i in range(len(traj))]
        cornerList=[self.isInCorner(self.spaceSize-self.cornerSize,coor) for coor in sheepCoordinateList]
        cornerNumber=len(np.where(cornerList)[0])
        cornerRatio = cornerNumber/ len(traj)
        return cornerRatio     

def isInCorner(lowerBound,coordinate):
	return np.all(np.abs(coordinate)>lowerBound)

class CountCollision :
    def __init__(self,sheepId,wolfId,stateIndex, positionIndex,collisionRadius,isCollision):
        self.sheepId =sheepId
        self.wolfId=wolfId 
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex
        self.isCollision=isCollision
        self.collisionRadius=collisionRadius
    def __call__(self,traj):
        sheepCoordinateList = [traj[i][self.stateIndex][self.sheepId][self.positionIndex] for i in range(len(traj))]
        wolfCoordinateList = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] for i in range(len(traj))]
        collisionList= [self.isCollision(coor1,coor2,self.collisionRadius) for (coor1, coor2) in zip(sheepCoordinateList, wolfCoordinateList)]

        collisionNumber = len(np.where(collisionList)[0])
        collisionRatio = collisionNumber / len(traj)
        return collisionRatio

def isCollision (coor1,coor2,disttance):
	return np.linalg.norm(coor1-coor2) < disttance   

if __name__ =='__main__' :

	dataFileDir = '../data'	

	# trajectories=pd.read_pickle(os.path.join(dataFileDir, '3.pickle'))
	trajectories=pd.read_pickle(os.path.join(dataFileDir, 'agentId=310_killzoneRadius=0.5_maxRunningSteps=250_numSimulations=300_offset=0.pickle'))
	x=[0.5,4]
	y=[0.8,2]
	print(isCrossAxis(x,y))
	sheepId=0
	wolfId=1
	masterId=2
	stateIndex=0
	positionIndex=[0,1]
	timeWindow=6
	maxCircleNumber=3
