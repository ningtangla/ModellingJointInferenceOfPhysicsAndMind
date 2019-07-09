import pandas as pd
import pygame

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
        # mindsPhysicsActionsDf['jointLikelihood'] = [getLikelihood(mind, physics, action) for index, value in mindsPhysicsActionsDf.iterrows() for mind, physics, action in index]

        mindsPhysicsActionsDf['jointLikelihood'] = [getLikelihood(index[0], index[1], index[2])
                                                    for index, value in mindsPhysicsActionsDf.iterrows()]

        actionsIntegratedOut = list(mindsPhysicsActionsDf.groupby([self.mindName, self.physicsName])[
            'jointLikelihood'].transform('sum'))

        priorLikelihoodPair = zip(mindsPhysicsPrior, actionsIntegratedOut)
        posteriorUnnormalized = [prior * likelihood for prior, likelihood in priorLikelihoodPair]
        unnormalizedSum = sum(posteriorUnnormalized)
        mindsPhysicsPosterior = [posterior / unnormalizedSum for posterior in posteriorUnnormalized]

        return mindsPhysicsPosterior


def saveImage(screenShotIndex, game):
    pygame.image.save(game, "screenshot" + format(screenShotIndex, '04') + ".png")


class Observe:
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __call__(self, timeStep):
        currentState = self.trajectory[timeStep]
        if timeStep >= len(self.trajectory):
            currentState = None
        return currentState


class InferDiscreteChasingAndDrawDemo:
    def __init__(self, isInferenceTerminal, observe, inferOneStep,
                 visualize = None, saveImage = None):
        self.isInferenceTerminal = isInferenceTerminal
        self.observe = observe
        self.inferOneStep = inferOneStep
        self.visualize = visualize
        self.saveImage = saveImage

    def __call__(self, mindsPhysicsPrior):
        currentState = self.observe(0)
        nextTimeStep = 1
        while True:
            print('round', nextTimeStep)
            if self.visualize:
                game = self.visualize(currentState, mindsPhysicsPrior)
                if self.saveImage is not None:
                    self.saveImage(nextTimeStep, game)
            nextState = self.observe(nextTimeStep)
            if nextState is None:
                return mindsPhysicsPrior
            mindsPhysicsPosterior = self.inferOneStep(currentState, nextState,
                                                 mindsPhysicsPrior)
            if self.isInferenceTerminal(mindsPhysicsPosterior):
                return mindsPhysicsPosterior
            currentState = nextState
            mindsPhysicsPrior = mindsPhysicsPosterior
            nextTimeStep += 1


