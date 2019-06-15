import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def makeTitle(xlabel, ylabel):
	return '%s vs %s' % (xlabel, ylabel)


def splitDictionary(originalDict, splitFactor):
	newDict = [{key: value[index] for key, value in originalDict.items()} for index in range(splitFactor)]
	return newDict


def dictToDataframe(data, axisName, lineVariableIndex):
	numOfDependentVariable = len(list(data.values())[0])
	splitedDependentVaraibles = splitDictionary(data, numOfDependentVariable)
	dataDFs = [pd.Series(dictionary).rename_axis(axisName).unstack(level=lineVariableIndex) for dictionary in splitedDependentVaraibles]
	return dataDFs


def drawPerGraph(dataDF, title):
	plt.title(title)
	dataDF.plot(title=title)
	plt.savefig(title)


def draw(data, independetVariablesName, graphValueName, lineVariableIndex=0, xVariableIndex=1):
	dataDFs = dictToDataframe(data, independetVariablesName, lineVariableIndex)
	plt.figure()
	titles = [makeTitle(graphValueName[index], independetVariablesName[xVariableIndex]) for index in range(len(dataDFs))]
	[drawPerGraph(dataDF, title) for dataDF, title in zip(dataDFs, titles)]


if __name__ == '__main__':
	# data = {(200,2): [10, -20], (200,4): [20,30], (200,8): [30,40], (200,16):[40,50],
	# 		(400,2): [15,25], (400,4): [25,35], (400,8): [35,45], (400,16): [45,55],
	# 		(800,2): [18,28], (800,4): [28,38], (800,8): [38,48], (800,16): [48,58],
	# 		(1000,2): [22,32], (1000,4): [32,42], (1000,8): [42,52], (1000,16): [52,62]}
	data = {('Train', 20): (1.369002, 0.9), ('Train', 40): (1.4647734, 0.8), ('Train', 80): (1.4684832, 0.8),
	        ('Train', 160): (1.5762081, 0.6875), ('Train', 320): (1.3548043, 0.915625), ('Test', 20): (1.8174871, 0.4526),
	        ('Test', 40): (1.7337555, 0.5356), ('Test', 80): (1.6390549, 0.6348), ('Test', 160): (1.6387422, 0.632),
	        ('Test', 320): (1.3977793, 0.8752)}
