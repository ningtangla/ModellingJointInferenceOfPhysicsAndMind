
import functools as ft
import numpy as np

class ApproximatePolicy():
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
        self.numActionSpace = len(self.actionSpace)
    def __call__(self, stateBatch, model):
        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        actionDistribution_ = graph.get_tensor_by_name('outputs/actionDistribution_:0')
        actionDistributionBatch = model.run(actionDistribution_, feed_dict = {state_ : stateBatch})
        actionIndexBatch = [np.random.choice(range(self.numActionSpace), p = actionDistribution) for actionDistribution in actionDistributionBatch]
        actionBatch = np.array([self.actionSpace[actionIndex] for actionIndex in actionIndexBatch])
        return actionBatch


class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal

    def __call__(self, actor):
        oldState, action = None, None
        oldState = self.transitionFunction(oldState, action)
        trajectory = []

        for time in range(self.maxTimeStep):
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = actor(oldStateBatch)
            action = actionBatch[0]
            # actionBatch shape: batch * action Dimension; only keep action Dimention in shape
            newState = self.transitionFunction(oldState, action)
            trajectory.append((oldState, action))
            terminal = self.isTerminal(oldState)
            if terminal:
                break
            oldState = newState
        return trajectory

class AccumulateRewards():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array([ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))])
        return accumulatedRewards

def normalize(accumulatedRewards):
    if np.std(accumulatedRewards) != 0:
        normalizedAccumulatedRewards = (accumulatedRewards - np.mean(accumulatedRewards)) / np.std(accumulatedRewards)
    else:
        return accumulatedRewards
    return normalizedAccumulatedRewards

class TrainTensorflow():
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
        self.numActionSpace = len(actionSpace)
    def __call__(self, episode, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))

        actionIndexEpisode = np.array([list(self.actionSpace).index(list(action)) for action in actionEpisode])
        actionLabelEpisode = np.zeros([numBatch, self.numActionSpace])
        actionLabelEpisode[np.arange(numBatch), actionIndexEpisode] = 1
        stateBatch, actionLabelBatch = np.array(stateEpisode).reshape(numBatch, -1), np.array(actionLabelEpisode).reshape(numBatch, -1)
        mergedAccumulatedRewardsEpisode = np.concatenate(normalizedAccumulatedRewardsEpisode)

        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        actionLabel_ = graph.get_tensor_by_name('inputs/actionLabel_:0')
        accumulatedRewards_ = graph.get_tensor_by_name('inputs/accumulatedRewards_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = model.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                    actionLabel_ : actionLabelBatch,
                                                                    accumulatedRewards_ : mergedAccumulatedRewardsEpisode
                                                                    })
        return loss, model

class PolicyGradient():
    def __init__(self, numTrajectory, maxEpisode, render):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
        self.render = render

    def __call__(self, model, approximatePolicy, sampleTrajectory, accumulateRewards, train):
        for episodeIndex in range(self.maxEpisode):
            print("Training episode", episodeIndex)
            policy = lambda state: approximatePolicy(state, model)
            episode = [sampleTrajectory(policy) for index in range(self.numTrajectory)]
            normalizedAccumulatedRewardsEpisode = [normalize(accumulateRewards(trajectory)) for trajectory in episode]
            loss, model = train(episode, normalizedAccumulatedRewardsEpisode, model)
            # print(np.mean([len(trajectory) for trajectory in episode]))
        return model