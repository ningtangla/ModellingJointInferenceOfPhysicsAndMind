# cd ~/ModellingJointInferenceOfPhysicsAndMind/data/generateExpDemo/trajectories/agentId=0_killzoneRadius=2_maxRunningSteps=100_numSimulations=200_offset=0
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
#         done
#     done
# done
# cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideosForLeashedDemo1.05

cd ~/ModellingJointInferenceOfPhysicsAndMind/data/generateExpDemo/trajectories
mkdir demo

for pureMCTSAgentId in 210
do

    for beta in 0.5 
    do
        for masterPowerRatio in 0.4
        do 
            for index in {0..27} 
            do
              
            ffmpeg -r 60 -f image2 -s 1920x1080 -i ~/ModellingJointInferenceOfPhysicsAndMind/data/generateMCTSDemo/image/beta=${beta}_index=${index}_killzoneRadius=0.5_masterPowerRatio=${masterPowerRatio}_maxRunningSteps=10_numAgents=3_numSimulations=80_pureMCTSAgentId=${pureMCTSAgentId}.pickle/image/%0d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/generateMCTSDemo/demo/beta=${beta}_index=${index}_killzoneRadius=0.5_masterPowerRatio=${masterPowerRatio}_maxRunningSteps=10_numAgents=3_numSimulations=80_pureMCTSAgentId=${pureMCTSAgentId}_Demo${index}.mp4

            done
        done
    done
done


cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/generateExpDemo


# cd ~/ModellingJointInferenceOfPhysicsAndMind/data/generateExpDemo/trajectories
# mkdir demo


# for offset in 0 
# do
#     for index in 0 
#     do

#     cd ~/ModellingJointInferenceOfPhysicsAndMind/data/generateExpDemo/trajectories/killzoneRadius=0.5_maxRunningSteps=90_numSimulations=400_offset=${offset}/${index}/condition=0

#     ffmpeg -r 40 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/generateExpDemo/trajectories/demo/killzoneRadius=0.5_maxRunningSteps=90_numSimulations=400_offset=${offset}_Demo${index}.mp4


#     done
# done

# cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/generateExpDemo




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
