cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchMasterPolicy/mctsSheep/
mkdir demo

for masterPowerRatio in 0.4 0.8
do
    for numSimulationsPerTree in 50 100
    do
        for predatorPowerRatio in 1.5 2
        do
            for index in 0 1 2
            do
            cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchMasterPolicy/mctsSheep/masterPowerRatio=${masterPowerRatio}_numSimulationsPerTree=${numSimulationsPerTree}_predatorPowerRatio=${predatorPowerRatio}/${index}

            ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/searchMasterPolicy/mctsSheep/demo/masterPowerRatio=${masterPowerRatio}_numSimulationsPerTree=${numSimulationsPerTree}_predatorPowerRatio=${predatorPowerRatio}_Demo${index}.mp4
            done
        done
    done
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