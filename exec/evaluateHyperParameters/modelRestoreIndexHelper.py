from collections import OrderedDict
import pandas as pd

def main():
	manipulatedHyperVariables = OrderedDict()
	manipulatedHyperVariables['miniBatchSize'] = [64, 32]  # [64, 128, 256]
	manipulatedHyperVariables['learningRate'] = [1e-3, 1e-5]  # [1e-2, 1e-3, 1e-4]
	manipulatedHyperVariables['numSimulations'] = [5,10] #[50, 100, 200]

	#numSimulations = manipulatedHyperVariables['numSimulations']
	levelNames = list(manipulatedHyperVariables.keys())
	levelValues = list(manipulatedHyperVariables.values())
	modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

	hyperVariablesConditionlist=[]
	hyperVariablesConditionlist=[{levelName:str(modelIndex.get_level_values(levelName)[modelIndexNumber]) for levelName in levelNames} for modelIndexNumber in range(len(modelIndex))]
	#add restore iterationIndex
	#print(hyperVariablesConditionlist)

	indexList=range(len(hyperVariablesConditionlist))
	[oneCondition.update({'index':str(indexList[i])}) for (i,oneCondition) in enumerate(hyperVariablesConditionlist) ]

	for oneCondition in hyperVariablesConditionlist:
		print(oneCondition)

	# restoreList=[0,0,0,100,0,0,0,0,0,0]
	# hyperVariablesConditionlist=[oneCondition(iterationIndex=restore) for oneCondition in hyperVariablesConditionlist]
if __name__ == '__main__':
    main()