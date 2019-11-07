import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, backup, InitializeChildren  # Expand, RollOut
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors

from src.episode import chooseGreedyAction  # SampleTrajectory
from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset  # IsTerminal

import timee
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle


class IsTerminal():
    def __init__(self, getPredatorPos, getPreyPos, minDistance, divideDegree):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.minDistance = minDistance
        self.divideDegree = divideDegree

    def __call__(self, lastState, currentState):
        terminal = False

        getPositionList = lambda getPos, lastState, currentState: np.linspace(getPos(lastState), getPos(currentState), self.divideDegree, endpoint=True)

        getL2Normdistance = lambda preyPosition, predatorPosition: np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2)

        preyPositionList = getPositionList(self.getPreyPos, lastState, currentState)
        predatorPositionList = getPositionList(self.getPredatorPos, lastState, currentState)

        L2NormdistanceList = [getL2Normdistance(preyPosition, predatorPosition) for (preyPosition, predatorPosition) in zip(preyPositionList, predatorPositionList)]

        if np.any(np.array(L2NormdistanceList) <= self.minDistance):
            terminal = True
        return terminal


class RewardFunctionCompete():
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal

    def __call__(self, lastState, currentState, action):
        reward = self.aliveBonus
        if self.isTerminal(lastState, currentState):
            reward += self.deathPenalty
        return reward


class RollOut:
    def __init__(self, rolloutPolicy, maxRolloutStep, transitionFunction, rewardFunction, isTerminal, rolloutHeuristic):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction
        self.maxRolloutStep = maxRolloutStep
        self.rolloutPolicy = rolloutPolicy
        self.isTerminal = isTerminal
        self.rolloutHeuristic = rolloutHeuristic

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        totalRewardForRollout = 0

        if leafNode.is_root:
            lastState = currentState
        else:
            lastState = list(leafNode.parent.id.values())[0]

        for rolloutStep in range(self.maxRolloutStep):
            action = self.rolloutPolicy(currentState)
            totalRewardForRollout += self.rewardFunction(lastState, currentState, action)
            if self.isTerminal(lastState, currentState):
                break
            nextState = self.transitionFunction(currentState, action)
            lastState = currentState
            currentState = nextState

        heuristicReward = 0
        if not self.isTerminal(lastState, currentState):
            heuristicReward = self.rolloutHeuristic(currentState)
        totalRewardForRollout += heuristicReward

        return totalRewardForRollout


class Expand:
    def __init__(self, isTerminal, initializeChildren):
        self.isTerminal = isTerminal
        self.initializeChildren = initializeChildren

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        if leafNode.is_root:
            lastState = currentState
        else:
            lastState = list(leafNode.parent.id.values())[0]
        if not self.isTerminal(lastState, currentState):
            leafNode.isExpanded = True
            leafNode = self.initializeChildren(leafNode)

        return leafNode


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state, state):
            state = self.reset()
            print(L2NormdistanceList)
            print(np.array(L2NormdistanceList) <= self.minDistance)

        trajectory = []
        lastState = state
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(lastState, state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            lastState = state
            state = nextState

        return trajectory


class SampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction
        self.render = render
        self.renderOn = renderOn

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state, state):
            state = self.reset()

        trajectory = []
        lastState = state
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(lastState, state):
                trajectory.append((state, None, None))
                break
            if self.renderOn:
                self.render(state)
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            lastState = state
            state = nextState

        return trajectory


if __name__ == "__main__":
    main()
