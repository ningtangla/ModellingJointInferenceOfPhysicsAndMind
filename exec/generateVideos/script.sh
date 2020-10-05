
for index in {10..17}
do

cd ~/Documents/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideos/demo/2ObjectsObstaclesNNGuideMCTS_selfIteration=6500_otherIteration=6500/${index}/

ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/Documents/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideos/demoNNGuideMCTS/selfIteration=6500_otherIteration=6500_selfId=0_Demo${index}.mp4

done

cd ~/Documents/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideos