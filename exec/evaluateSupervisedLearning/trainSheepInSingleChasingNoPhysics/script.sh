cd ~/ModellingJointInferenceOfPhysicsAndMind/data/demoImg/

for killzoneRadius in 10 15 20 30 40 45 60
do
    for preyPowerRatio in 1.5 3 4.5
    do
        for predatorPowerRatio in 1 2 3
        do
        cd ~/ModellingJointInferenceOfPhysicsAndMind/data/demoImg/preyPowerRatio=${preyPowerRatio}_predatorPowerRatio=${predatorPowerRatio}_killzoneRadius=${killzoneRadius}

        ffmpeg -r 10 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/demo/preyPowerRatio=${preyPowerRatio}_predatorPowerRatio=${predatorPowerRatio}_killzoneRadius=${killzoneRadius}.mp4
        # mv *.Demo${index}.mp4 ../../demo
        done
    done
done
cd ~/ModellingJointInferenceOfPhysicsAndMind/exec/evaluateSupervisedLearning/trainSheepInSingleChasingNoPhysics

