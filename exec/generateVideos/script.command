cd /Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideos/demo/0
mkdir demoPics
mv 0* demoPics
cd demoPics
ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p Demo.mp4
mv Demo.mp4 ..

cd /Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideos/demo/1
mkdir demoPics
mv 0* demoPics
cd demoPics
ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p Demo.mp4
mv Demo.mp4 ..

cd /Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideos/demo/2
mkdir demoPics
mv 0* demoPics
cd demoPics
ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p Demo.mp4
mv Demo.mp4 ..

cd /Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideos/demo/3
mkdir demoPics
mv 0* demoPics
cd demoPics
ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p Demo.mp4
mv Demo.mp4 ..

cd /Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/exec/generateVideos/demo/4
mkdir demoPics
mv 0* demoPics
cd demoPics
ffmpeg -r 60 -f image2 -s 1920x1080 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p Demo.mp4
mv Demo.mp4 ..
