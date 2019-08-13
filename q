[1mdiff --cc exec/evaluateSupervisedLearning/prepareLeasedSingleMCTSAgentDataParallel.py[m
[1mindex caf34f0,d418a07..0000000[m
[1m--- a/exec/evaluateSupervisedLearning/prepareLeasedSingleMCTSAgentDataParallel.py[m
[1m+++ b/exec/evaluateSupervisedLearning/prepareLeasedSingleMCTSAgentDataParallel.py[m
[36m@@@ -44,10 -44,9 +44,10 @@@[m [mdef main()[m
  [m
      startTime = time.time()[m
  [m
[31m-     numTrajectories = 5000[m
[32m+     numTrajectories = 4000[m
      # generate and load trajectories before train parallelly[m
[31m -    sampleTrajectoryFileName = 'sampleMCTSLeasedWolfTrajectory.py'[m
[32m +    sampleTrajectoryFileName = 'sampleMCTSSheepTrajectoryWithNNWolf.py'[m
[32m +[m
      # sampleTrajectoryFileName = 'sampleMCTSSheepTrajectory.py'[m
      numCpuCores = os.cpu_count()[m
      print(numCpuCores)[m
[36m@@@ -73,9 -72,9 +73,9 @@@[m
  [m
          cmdList = generateTrajectoriesParallel(pathParameters)[m
          # print(cmdList)[m
[31m -        trajectories = loadTrajectoriesForParallel(pathParameters)[m
[32m +        # trajectories = loadTrajectoriesForParallel(pathParameters)[m
          # import ipdb; ipdb.set_trace()[m
[31m-     [m
[32m+ [m
      endTime = time.time()[m
      print("Time taken {} seconds".format((endTime - startTime)))[m
  [m
