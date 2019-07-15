cd /Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/evaluateNumTerminalTrajectoriesWolfChaseSheepMCTSRolloutMujoco/demo/videosNumSim50/9
mkdir demoPics
mv 0* demoPics
cd demoPics
ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p Demo.mp4
mv Demo.mp4 ..