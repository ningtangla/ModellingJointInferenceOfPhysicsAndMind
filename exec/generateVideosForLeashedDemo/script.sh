# cd ~/ModellingJointInferenceOfPhysicsAndMind/data/evaluateSupervisedLearning/evaluateTrajectories/agentId=1_maxRunningSteps=20_numSimulations=100/
# mkdir demo
# cd 0
# ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p Demo0.mp4
# mv Demo0.mp4 ../demo


cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/
mkdir demo

for draggerMass in 10 13
do
    for maxTendonLength in  0.3 0.6
    do
        for predatorMass in 10 13
        do
            for tendonDamping in 0.3 0.6
            do
                for tendonStiffness in 5 10
                do
                    for index in 0
                    do
                        mkdir ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/draggerMass=${draggerMass}_maxTendonLength=${maxTendonLength}_predatorMass=${predatorMass}_tendonDamping=${tendonDamping}_tendonStiffness=${tendonStiffness}/demo

                        cd ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/draggerMass=${draggerMass}_maxTendonLength=${maxTendonLength}_predatorMass=${predatorMass}_tendonDamping=${tendonDamping}_tendonStiffness=${tendonStiffness}/${index}

                        ffmpeg -r  60 -f image2 -s 1920x1080 -i  %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/searchLeashedModelParameters/leasedTrajectories/demo/draggerMass=${draggerMass}_maxTendonLength=${maxTendonLength}_predatorMass=${predatorMass}_tendonDamping=${tendonDamping}_tendonStiffness=${tendonStiffness}_sampleIndex${index}.mp4
                    done
                done
            done
        done
    done
done

cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideosForLeashedDemo
