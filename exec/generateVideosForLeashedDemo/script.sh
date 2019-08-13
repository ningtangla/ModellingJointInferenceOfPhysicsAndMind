# cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchToWallHerustic/mctsSheep/
# mkdir demo

# for heuristicWeightWallDis in 0.5 1 2
# do
#     for preyPowerRatio in 0.4 0.6 0.8
#     do
#         for index in 1 2
#         do
#         cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchToWallHerustic/mctsSheep/heuristicWeightWallDis=${heuristicWeightWallDis}_preyPowerRatio=${preyPowerRatio}
#         cd ${index}
#         ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/searchToWallHerustic/mctsSheep/demo/heuristicWeightWallDis=${heuristicWeightWallDis}_preyPowerRatio=${preyPowerRatio}_Demo${index}.mp4
#         # mv *.Demo${index}.mp4 ../../demo
#         done
#     done
# done
# cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideosForLeashedDemo
# cd ~/ModellingJointInferenceOfPhysicsAndMind/data/agentId=0_depth=8_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=128_numSimulations=100_trainSteps=350000/
# mkdir demo


cd ~/ModellingJointInferenceOfPhysicsAndMind/data/evaluateSupervisedLearning/leashedDistractorTrajectories/agentId=3_killzoneRadius=1_maxRunningSteps=60_numSimulations=100
mkdir demo

for index in 0 1 2 3 4 5 6 7 8 9
do
cd ~/ModellingJointInferenceOfPhysicsAndMind/data/evaluateSupervisedLearning/leashedDistractorTrajectories/agentId=3_killzoneRadius=1_maxRunningSteps=60_numSimulations=100/${index}

ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/evaluateSupervisedLearning/leashedDistractorTrajectories/agentId=3_killzoneRadius=1_maxRunningSteps=60_numSimulations=100/demo/agentId=3_killzoneRadius=1_maxRunningSteps=60_numSimulations=100_Demo${index}.mp4
# mv *.Demo${index}.mp4 ../../demo
done

cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideosForLeashedDemo

# cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/
# mkdir demo

# for draggerMass in 8 10 12
# do
#     for maxTendonLength in  0.6
#     do
#         for predatorMass in 10
#         do
#             for tendonDamping in 0.7
#             do
#                 for tendonStiffness in 10
#                 do
#                     for index in 0 1 2
#                     do
#                         mkdir ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/draggerMass=${draggerMass}_maxTendonLength=${maxTendonLength}_predatorMass=${predatorMass}_tendonDamping=${tendonDamping}_tendonStiffness=${tendonStiffness}/demo


#                             cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/draggerMass=${draggerMass}_maxTendonLength=${maxTendonLength}_predatorMass=${predatorMass}_predatorPower=${predatorPower}_tendonDamping=${tendonDamping}_tendonStiffness=${tendonStiffness}/${index}

#                             ffmpeg -r  60 -f image2 -s 1920x1080 -i  %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/demo/draggerMass=${draggerMass}_maxTendonLength=${maxTendonLength}_predatorMass=${predatorMass}_predatorPower=${predatorPower}_tendonDamping=${tendonDamping}_tendonStiffness=${tendonStiffness}_sampleIndex${index}.mp4
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideosForLeashedDemo
