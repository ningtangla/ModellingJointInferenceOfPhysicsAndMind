import pandas as pd
import pygame as pg
import numpy as np

def softenPolicy(policy):
    getNormalizingParam = lambda actionDist: np.sum(np.exp(list(actionDist.values())))
    softenActionProb = lambda actionDist: {action: np.exp(actionDist[action]) / getNormalizingParam(actionDist) for action
                                     in actionDist.keys()}
    softenedPolicy = lambda state: softenActionProb(policy(state))

    return softenedPolicy



class ObserveStateOnly:
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __call__(self, timeStep):
        if timeStep >= len(self.trajectory):
            return None
        currentState = self.trajectory[timeStep]
        return currentState


class Observe:
    def __init__(self, stateIndex, trajectory):
        self.stateIndex = stateIndex
        self.trajectory = trajectory

    def __call__(self, timeStep):
        if timeStep >= len(self.trajectory):
            return None
        currentState = self.trajectory[timeStep][self.stateIndex]
        return currentState


class ObserveWithRope:
    def __init__(self, stateIndex, trajectory, sheepWolfMasterIndex = 3):
        self.stateIndex = stateIndex
        self.trajectory = trajectory
        self.sheepWolfMasterIndex = sheepWolfMasterIndex

    def __call__(self, timeStep):
        if timeStep >= len(self.trajectory):
            return None
        currentState = self.trajectory[timeStep][0][self.stateIndex][:self.sheepWolfMasterIndex]
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
    def __init__(self, inferenceIndex, mindPhysicsName, getMindsPhysicsActionsJointLikelihood):
        self.inferenceIndex = inferenceIndex
        self.mindName, self.physicsName = mindPhysicsName
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


class InferOneStepLikelihood:
    def __init__(self, inferenceIndex, getMindsPhysicsActionsJointLikelihood):
        self.inferenceIndex = inferenceIndex
        self.getMindsPhysicsActionsJointLikelihood = getMindsPhysicsActionsJointLikelihood

    def __call__(self, state, nextState):
        mindsPhysicsActionsDf = pd.DataFrame(index=self.inferenceIndex)
        getLikelihood = lambda mind, physics, action: self.getMindsPhysicsActionsJointLikelihood(mind, state, action,
                                                                                                 physics, nextState)

        jointLikelihood = [getLikelihood(index[0], index[1], index[2]) for index, value in mindsPhysicsActionsDf.iterrows()]
        return jointLikelihood


class QueryDecayedLikelihood:
    def __init__(self, mindPhysicsName, decayParameter):
        self.mindName, self.physicsName = mindPhysicsName
        self.decayParam = decayParameter

    def __call__(self, likelihoodDf, queryTimeStep):

        posteriorDf = pd.DataFrame(index = likelihoodDf.index)

        for timeStep in range(queryTimeStep+1):
            actionsIntegratedOut = list(likelihoodDf.groupby([self.mindName, self.physicsName])[
                timeStep].transform('sum'))
            posteriorDf[timeStep] = np.log(actionsIntegratedOut)

        queryLogLikelihood = np.sum([posteriorDf[timeStep] * np.power(self.decayParam, queryTimeStep - timeStep)
                                  for timeStep in range(queryTimeStep+ 1)], axis = 0)

        queryLikelihood = np.exp(queryLogLikelihood) / np.sum(np.exp(queryLogLikelihood))

        # query1LogLikelihood = np.sum([posteriorDf[timeStep] for timeStep in range(queryTimeStep+ 1)], axis = 0)
        #
        # query1Likelihood = np.exp(query1LogLikelihood) / np.sum(np.exp(query1LogLikelihood))

        # print('nodecayLog', query1LogLikelihood)
        # print('decayLog', queryLogLikelihood)
        #
        # print('nodecayexp', np.exp(query1LogLikelihood))
        # print('decayexp', np.exp(queryLogLikelihood))
        #
        # print('nodecayexpSum', np.sum(np.exp(query1LogLikelihood)))
        # print('decayexpSum', np.sum(np.exp(queryLogLikelihood)))
        #
        # print('param', self.decayParam)
        #
        # print('nodecay', query1Likelihood)
        # print('decay', queryLikelihood)

        return queryLikelihood



class InferDiscreteChasingWithMemoryDecayAndDrawDemo:
    def __init__(self, fps, inferenceIndex, isInferenceTerminal, observe, inferOneStepLik, queryLikelihood,
                 visualize = None, saveImage = None):
        self.fps = fps
        self.inferenceIndex = inferenceIndex

        self.isInferenceTerminal = isInferenceTerminal
        self.observe = observe
        self.inferOneStepLik = inferOneStepLik
        self.queryLikelihood = queryLikelihood

        self.visualize = visualize
        self.saveImage = saveImage

    def __call__(self, mindsPhysicsPrior):
        currentTimeStep = 0
        mindsPhysicsCurrentLikelihood = mindsPhysicsPrior

        inferenceLikDf = pd.DataFrame(index = self.inferenceIndex)
        queriedLikelihoodDf = pd.DataFrame(index = self.inferenceIndex)
        fpsClock = pg.time.Clock()

        while True:
            print('round', currentTimeStep)

            inferenceLikDf[currentTimeStep] = mindsPhysicsCurrentLikelihood
            # currentLikelihood of timestep 0 = inference result of timestep 0 and timestep 1
            # print(inferenceLikDf)
            # print(np.sum(mindsPhysicsCurrentLikelihood))

            currentPosterior = self.queryLikelihood(inferenceLikDf, currentTimeStep)
            queriedLikelihoodDf[currentTimeStep] = currentPosterior

            if self.isInferenceTerminal(currentPosterior):
                return queriedLikelihoodDf

            currentState = self.observe(currentTimeStep)
            if self.visualize:
                fpsClock.tick(self.fps)
                game = self.visualize(currentState, currentPosterior)
                if self.saveImage is not None:
                    self.saveImage(game)

            nextTimeStep = currentTimeStep + 1
            nextState = self.observe(nextTimeStep)

            if nextState is None:
                return queriedLikelihoodDf

            mindsPhysicsCurrentLikelihood = self.inferOneStepLik(currentState, nextState)
            currentTimeStep = nextTimeStep




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
                    self.saveImage(game)

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
    def __init__(self, fps, inferenceIndex, isInferenceTerminal, observe, queryLikelihood, inferOneStepLik, visualize = None):
        self.fps = fps
        self.inferenceIndex = inferenceIndex

        self.isInferenceTerminal = isInferenceTerminal
        self.observe = observe
        self.queryLikelihood = queryLikelihood
        self.inferOneStepLik = inferOneStepLik

        self.visualize = visualize

    def __call__(self, numOfAgents, mindsPhysicsPrior):
        currentTimeStep = 0
        mindsPhysicsCurrentLikelihood = mindsPhysicsPrior

        inferenceLikDf = pd.DataFrame(index = self.inferenceIndex)
        queriedLikelihoodDf = pd.DataFrame(index = self.inferenceIndex)
        fpsClock = pg.time.Clock()

        while True:
            print('round', currentTimeStep)
            inferenceLikDf[currentTimeStep] = mindsPhysicsCurrentLikelihood
            currentPosterior = self.queryLikelihood(inferenceLikDf, currentTimeStep)
            queriedLikelihoodDf[currentTimeStep] = currentPosterior

            if self.isInferenceTerminal(currentPosterior):
                return queriedLikelihoodDf

            currentState = self.observe(currentTimeStep)
            nextTimeStep = currentTimeStep + 1
            nextState = self.observe(nextTimeStep)

            if nextState is None:
                return queriedLikelihoodDf

            if self.visualize:
                fpsClock.tick(self.fps)
                agentsCurrentState = currentState[:numOfAgents]
                agentsNextState = nextState[:numOfAgents]
                self.visualize(agentsCurrentState, agentsNextState, currentPosterior)

            mindsPhysicsCurrentLikelihood = self.inferOneStepLik(currentState, nextState)
            currentTimeStep = nextTimeStep



class InferContinuousChasingAndDrawDemoNoDecay:
    def __init__(self, fps, inferenceIndex, isInferenceTerminal, observe, inferOneStep, visualize = None):
        self.fps = fps
        self.inferenceIndex = inferenceIndex

        self.isInferenceTerminal = isInferenceTerminal
        self.observe = observe
        self.inferOneStep = inferOneStep
        self.visualize = visualize

    def __call__(self, numOfAgents, mindsPhysicsPrior):
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
                agentsCurrentState = currentState[:numOfAgents]
                agentsNextState = nextState[:numOfAgents]
                self.visualize(agentsCurrentState, agentsNextState, mindsPhysicsPrior)

            mindsPhysicsPosterior = self.inferOneStep(currentState, nextState, mindsPhysicsPrior)

            nextTimeStep += 1
            mindsPhysicsPrior = mindsPhysicsPosterior
            currentState = nextState

            if self.isInferenceTerminal(mindsPhysicsPosterior):
                mindsPhysicsActionsDf[nextTimeStep] = mindsPhysicsPrior
                return mindsPhysicsActionsDf

