cd ~/ModellingJointInferenceOfPhysicsAndMind/data/evaluateSupervisedLearning/leasedTrajectories/agentId=1_killzoneRadius=2_maxRunningSteps=25_numSimulations=100
mkdir demo
cd 0
ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p Demo0.mp4
mv Demo0.mp4 ../demo
cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideosForLeashedDemo


# cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/
# mkdir demo

<<<<<<< HEAD
for draggerMass in 8 10 12
do
    for maxTendonLength in  0.6
    do
        for predatorMass in 10
        do
            for tendonDamping in 0.7
            do
                for tendonStiffness in 10
                do
                    for index in 0 1 2
                    do
                        mkdir ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/draggerMass=${draggerMass}_maxTendonLength=${maxTendonLength}_predatorMass=${predatorMass}_tendonDamping=${tendonDamping}_tendonStiffness=${tendonStiffness}/demo
=======
# for draggerMass in 8 10 12
# do
#     for maxTendonLength in  0.6
#     do
#         for predatorMass in 10
#         do
#             for predatorPower in 1 1.3 1.6
#             do
#                 for tendonDamping in 0.7
#                 do
#                     for tendonStiffness in  10
#                     do
#                         for index in 0 1 2
#                         do
#                             mkdir ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/draggerMass=${draggerMass}_maxTendonLength=${maxTendonLength}_predatorMass=${predatorMass}_predatorPower=${predatorPower}_tendonDamping=${tendonDamping}_tendonStiffness=${tendonStiffness}/demo
>>>>>>> 40cbbbb77c98840ffb2e02ae812be2faa977d5cc

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
