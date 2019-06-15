import numpy as np


class Evaluate:
	def __init__(self, sampleTraj, numOfEpisodes):
		self.sampleTraj = sampleTraj
		self.numOfEpisodes = numOfEpisodes

	def __call__(self, policy):
		demoEpisode = [zip(*self.sampleTraj(policy)) for index in range(self.numOfEpisodes)]
		demoStates = [states for states, actions in demoEpisode]
		demoStateLengths = [len(traj) for traj in demoStates]
		avgLen = np.mean(demoStateLengths)
		stdLen = np.std(demoStateLengths)
		minLen = np.min(demoStateLengths)
		maxLen = np.max(demoStateLengths)
		medLen = np.median(demoStateLengths)
		return {"mean": avgLen, "var": stdLen, "min": minLen, "max": maxLen, "median": medLen}, demoStates
