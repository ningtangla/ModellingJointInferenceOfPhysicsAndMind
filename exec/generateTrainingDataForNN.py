import sys
sys.path.append('..')
import numpy as np
import pickle

import src.play as play


def distToGreedyAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedAction = actions[selectedIndex]
    return selectedAction


def distsToActions(distToAction, dists):
    actions = [distToAction(dist) if type(dist) is dict else dist for dist in dists]
    return actions


def main():
    pass


if __name__ == "__main__":
    main()
