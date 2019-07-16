# random policy: inference should not be able to infer anything
# noise to the transition -> make it deterministic first!
import pandas as pd
import pygame as pg

class Observe:
    def __init__(self, stateIndex, trajectory):
        self.stateIndex = stateIndex
        self.trajectory = trajectory

    def __call__(self, timeStep):
        if timeStep >= len(self.trajectory):
            return None
        currentState = self.trajectory[timeStep][self.stateIndex]
        return currentState


class IsInferenceTerminal:
    def __init__(self, thresholdPosterior, mindPhysicsName, inferenceIndex):
        self.thresholdPosterior = thresholdPosterior
        self.mindName, self.physicsName = mindPhysicsName
        self.inferenceIndex = inferenceIndex

    def __call__(self, posterior):
        inferenceDf = pd.DataFrame(index= self.inferenceIndex)
        inferenceDf['posterior'] = posterior
        rank = inferenceDf.groupby([self.mindName, self.physicsName]).sum().reset_index()
        largestPosterior = max(rank['posterior'])
        if largestPosterior >= self.thresholdPosterior:
            return True
        else:
            return False


class InferOneStep:
    def __init__(self, inferenceIndex, mindPhysicsActionName, getMindsPhysicsActionsJointLikelihood):
        self.inferenceIndex = inferenceIndex
        self.mindName, self.physicsName, self.actionName = mindPhysicsActionName
        self.getMindsPhysicsActionsJointLikelihood = getMindsPhysicsActionsJointLikelihood

    def __call__(self, state, nextState, mindsPhysicsPrior):
        mindsPhysicsActionsDf = pd.DataFrame(index = self.inferenceIndex)
        getLikelihood = lambda mind, physics, action: self.getMindsPhysicsActionsJointLikelihood(mind, state, action, physics, nextState)

        mindsPhysicsActionsDf['jointLikelihood'] = [getLikelihood(index[0], index[1], index[2])
                                                    for index, value in mindsPhysicsActionsDf.iterrows()]

        actionsIntegratedOut = list(mindsPhysicsActionsDf.groupby([self.mindName, self.physicsName])[
            'jointLikelihood'].transform('sum'))

        priorLikelihoodPair = zip(mindsPhysicsPrior, actionsIntegratedOut)
        posteriorUnnormalized = [prior * likelihood for prior, likelihood in priorLikelihoodPair]
        unnormalizedSum = sum(posteriorUnnormalized)
        mindsPhysicsPosterior = [posterior / unnormalizedSum for posterior in posteriorUnnormalized]

        return mindsPhysicsPosterior


class InferDiscreteChasingAndDrawDemo:
    def __init__(self, fps, inferenceIndex, isInferenceTerminal, observe, inferOneStep,
                 visualize = None, saveImage = None):
        self.fps = fps
        self.inferenceIndex = inferenceIndex

        self.isInferenceTerminal = isInferenceTerminal
        self.observe = observe
        self.inferOneStep = inferOneStep
        self.visualize = visualize
        self.saveImage = saveImage

    def __call__(self, mindsPhysicsPrior):
        currentState = self.observe(0)
        nextTimeStep = 1
        mindsPhysicsActionsDf = pd.DataFrame(index = self.inferenceIndex)
        fpsClock = pg.time.Clock()
        while True:
            mindsPhysicsActionsDf[nextTimeStep] = mindsPhysicsPrior
            print('round', nextTimeStep)
            if self.visualize:
                fpsClock.tick(self.fps)
                game = self.visualize(currentState, mindsPhysicsPrior)
                if self.saveImage is not None:
                    self.saveImage(nextTimeStep, game)

            nextState = self.observe(nextTimeStep)
            if nextState is None:
                return mindsPhysicsActionsDf
            mindsPhysicsPosterior = self.inferOneStep(currentState, nextState, mindsPhysicsPrior)

            nextTimeStep += 1
            mindsPhysicsPrior = mindsPhysicsPosterior
            currentState = nextState

            if self.isInferenceTerminal(mindsPhysicsPosterior):
                mindsPhysicsActionsDf[nextTimeStep] = mindsPhysicsPrior
                return mindsPhysicsActionsDf




class InferContinuousChasingAndDrawDemo:
    def __init__(self, fps, inferenceIndex, isInferenceTerminal, observe, inferOneStep, visualize = None):
        self.fps = fps
        self.inferenceIndex = inferenceIndex

        self.isInferenceTerminal = isInferenceTerminal
        self.observe = observe
        self.inferOneStep = inferOneStep
        self.visualize = visualize

    def __call__(self, mindsPhysicsPrior):
        currentState = self.observe(0)
        nextTimeStep = 1
        mindsPhysicsActionsDf = pd.DataFrame(index = self.inferenceIndex)
        fpsClock = pg.time.Clock()
        while True:
            mindsPhysicsActionsDf[nextTimeStep] = mindsPhysicsPrior
            print('round', nextTimeStep)
            nextState = self.observe(nextTimeStep)
            if nextState is None:
                return mindsPhysicsActionsDf

            if self.visualize:
                fpsClock.tick(self.fps)
                self.visualize(currentState, nextState, mindsPhysicsPrior)

            mindsPhysicsPosterior = self.inferOneStep(currentState, nextState, mindsPhysicsPrior)

            nextTimeStep += 1
            mindsPhysicsPrior = mindsPhysicsPosterior
            currentState = nextState

            if self.isInferenceTerminal(mindsPhysicsPosterior):
                mindsPhysicsActionsDf[nextTimeStep] = mindsPhysicsPrior
                return mindsPhysicsActionsDf



