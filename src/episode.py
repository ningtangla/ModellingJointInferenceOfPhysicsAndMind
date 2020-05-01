import numpy as np
import random
import pygame as pg
import os


class MultiAgentSampleTrajectory:
    def __init__(self, agentNames, iterationNumber, isTerminal, reset, currentState=None):
        self.agentNames = agentNames
        self.iterationNumber = iterationNumber
        self.isTerminal = isTerminal
        self.reset = reset
        self.currentState = currentState

    def __call__(self, multiAgentPolicy, multiAgentTransition):
        if self.currentState is None:
            self.currentState = self.reset()

        trajectory = [self.currentState]

        for i in range(self.iterationNumber):
            allAgentNextAction = multiAgentPolicy(self.currentState)
            nextState = multiAgentTransition(allAgentNextAction, self.currentState)
            trajectory.append(nextState)

            self.currentState = nextState
            if self.isTerminal(self.currentState):
                break
        return trajectory


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [choose(actionDist) for choose, actionDist in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class Sample3ObjectsTrajectory:
    def __init__(self, maxRunningSteps, transit, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()
        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class SampleTrajectoryTerminationProbability:
    def __init__(self, terminationProbability, transit, isTerminal, reset, chooseAction):
        self.terminationProbability = terminationProbability
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        terminal = False
        while(terminal == False):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState
            terminal = random.random() < self.terminationProbability

        return trajectory


def chooseGreedyAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedAction = actions[selectedIndex]
    return selectedAction


class SampleAction():
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        probs = list(actionDist.values())
        newProbs = np.array([np.power(prob, self.beta) for prob in probs])
        normProbs = newProbs / np.sum(newProbs)
        selectedIndex = list(np.random.multinomial(1, normProbs)).index(1)
        selectedAction = actions[selectedIndex]
        return selectedAction


class SelectSoftmaxAction():
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        probs = list(actionDist.values())

        exponent = np.multiply(probs, self.beta)
        newProbs = np.exp(exponent) / np.sum(np.exp(exponent))

        selectedIndex = list(np.random.multinomial(1, newProbs)).index(1)
        selectedAction = actions[selectedIndex]
        return selectedAction


class SampleAction():
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        probs = list(actionDist.values())
        newProbs = np.array([np.power(prob, self.beta) for prob in probs])
        normProbs = newProbs / np.sum(newProbs)
        selectedIndex = list(np.random.multinomial(1, normProbs)).index(1)
        selectedAction = actions[selectedIndex]
        return selectedAction


def getPairedTrajectory(agentsTrajectory):
    timeStepCount = len(agentsTrajectory[0])
    pairedTraj = [[agentTrajectory[timeStep] for agentTrajectory in agentsTrajectory] for timeStep in range(timeStepCount)]
    return pairedTraj


class Render():
    def __init__(self, numOfAgent, posIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir, drawBackground):
        self.numOfAgent = numOfAgent
        self.posIndex = posIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir
        self.drawBackground = drawBackground

    def __call__(self, state):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
            self.drawBackground()
            for i in range(self.numOfAgent):
                agentPos = state[i][self.posIndex]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(
                    agentPos[0]), np.int(agentPos[1])], self.circleSize)
            pg.display.flip()
            pg.time.wait(100)

            if self.saveImage == True:
                filenameList = os.listdir(self.saveImageDir)
                pg.image.save(self.screen, self.saveImageDir + '/' + str(len(filenameList)) + '.png')


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

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            if self.renderOn:
                self.render(state)
            actionDists = policy(state)
            action = [choose(action) for choose, action in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class SampleTrajectoryWithRenderWithInterpolationTerminal:
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
                self.render(state, runningStep)
            actionDists = policy(state)
            action = [choose(action) for choose, action in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            lastState = state
            state = nextState

        return trajectory
