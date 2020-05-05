
import os
import cv2
import time
 

class Pic2Video():

    def __init__(self,fps,size,fourcc,pictureLoadExtension='.png'):
        self.fps=fps
        self.size=size
        self.fourcc=fourcc
        self.pictureLoadExtension=pictureLoadExtension

    def __call__(self, picFolderPath,videoSavePath):

        fileList = os.listdir(picFolderPath)
        video = cv2.VideoWriter( videoPath, self.fourcc, self.fps, self.size )
        
        for item in fileList:
            if item.endswith(self.pictureLoadExtension):
                item = picFolderPath + '/' + item 
                img = cv2.imread(item) 
                video.write(img)
        video.release()
        print('video save at',videoSavePath)


def main():
    dirName = os.path.dirname(__file__)

    dataFolderName=os.path.join(dirName,'..','..', '..', 'data')
    mainImageDir = os.path.join(dataFolderName,'obstacaleDemoImg','test')

    videoSvaeDir = os.path.join(dataFolderName,'obstacaleDemoVideo','test')

    
    numSimulations=200
    maxRolloutSteps=30
    agentId=1
    maxRunningSteps = 30
    killzoneRadius = 2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius,'maxRolloutSteps':maxRolloutSteps}


    fps=12
    size=(1920,1080)
    videoSaveExtension='.mp4' 
    pictureLoadExtension='.png'
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    pic2Video=Pic2Video(fps,size,fourcc,pictureLoadExtension)

    generateVideoSavePath = GetSavePath(videoSvaeDir, videoSaveExtension, fixedParameters)



    loadTrajectoryIdList=range(8)
    for n  in loadTrajectoryIdList:
        loadImageDir=os.path.join(mainImageDir,str(n))
        videoSavePath=generateVideoSavePath({'Id':n})
        pic2Video(loadImageDir,videoSavePath)

if __name__ == "__main__":
    main()