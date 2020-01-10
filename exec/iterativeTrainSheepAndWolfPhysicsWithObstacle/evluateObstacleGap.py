import sys
import os
import mujoco_py as mujoco
import numpy as np
# from xml.dom.minidom import parse
from collections import OrderedDict
import xmltodict
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..','..'))
# import xmltodict
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
def parse_file(xml_path):
    '''
    Reads xml from xml_path, consolidates all includes in xml, and returns
    a normalized xml dictionary.  See preprocess()
    '''
    # TODO: use XSS or DTD checking to verify XML structure
    with open(xml_path) as f:
        xml_string = f.read()

    xml_doc_dict = xmltodict.parse(xml_string.strip())

    return xml_doc_dict

def transferNumberListToStr(numList):
    strList=[str(num) for num in numList]
    return ' '.join(strList)
def changeWallProperty(envDict,wallPropertyDict):
    for number,propertyDict in wallPropertyDict.items():
        for name,value in propertyDict.items():
            envDict['mujoco']['worldbody']['body'][number]['geom'][name]=value

    return envDict
def main():
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, 'twoAgentsTwoObstacles4.xml')
    xml_dict=parse_file(physicsDynamicsPath)

    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    xml_doc_dict = xmltodict.parse(xml_string.strip())

    wallPropertyDict={}
    wall1Id=5
    wall2Id=6
    gapLenth=1.65
    wall1Pos=[0,(9.95+gapLenth/2)/2,-0.2]
    wall1Size=[0.9,(9.95+gapLenth/2)/2-gapLenth/2,1.5]
    wall2Pos=[0,-(9.95+gapLenth/2)/2,-0.2]
    wall2Size=[0.9,(9.95+gapLenth/2)/2-gapLenth/2,1.5]

    wallPropertyDict[wall1Id]={'@pos':transferNumberListToStr(wall1Pos),'@size':transferNumberListToStr(wall1Size)}
    wallPropertyDict[wall2Id]={'@pos':transferNumberListToStr(wall2Pos),'@size':transferNumberListToStr(wall2Size)}

    xml_doc_dict=changeWallProperty(xml_doc_dict,wallPropertyDict)
    # xml_doc_dict['mujoco']['worldbody']['body'][5]['geom']['@pos']='0 3 -0.2'
    # xml_doc_dict['mujoco']['worldbody']['body'][5]['geom']['@size']='0.9 2 1.5'
    # xml_doc_dict['mujoco']['worldbody']['body'][5]['geom']


    # print(physicsModelXml)
    # name='obstacle11'
    # rootNode=physicsModelXml.documentElement
    # wall=rootNode.getElementsByTagName('body')
    xml=xmltodict.unparse(xml_doc_dict)
    print(xml_doc_dict['mujoco']['worldbody']['body'][5]['geom'])
    print(xml_doc_dict['mujoco']['worldbody']['body'][6]['geom'])
    # print(xml)
    # print(wall[6].toxml())
    # wall[6].childNodes[1]=geom condim="3" mass="10000" name="obstacle2" pos="0 -2 -0.2" size="0.5 1.75 1.5" type="box"
    # wall[6].childNodes[1]=,.
    # print(wall[6].childNodes[1].toxml())
    # print(wall[6].childNodes[1])
    physicsModel = mujoco.load_model_from_xml(xml)
    physicsSimulation = mujoco.MjSim(physicsModel)

    physicsSimulation.model.body_mass[8] = 30

    physicsSimulation.model.geom_friction[:,0] = 0.15

    physicsSimulation.set_constants()
    physicsSimulation.forward()

    # physicsSimulation.data.qpos[:] = np.array(init).flatten()


    qPos=np.array([-5.8, -5, 5, 0]).flatten()
    physicsSimulation.data.qpos[:] = qPos
    physicsSimulation.step()


    physicsViewer = mujoco.MjViewer(physicsSimulation)
    numSimulationFrames = 1500
    totalMaxVel = 0
    print(physicsSimulation.data.qvel, '!!!')
    print(physicsSimulation.data.qpos, '~~~')
    print(physicsSimulation.data.body_xpos, '...')
    for frameIndex in range(numSimulationFrames):
        if frameIndex == 550 or frameIndex == 600:
            print(physicsSimulation.data.ctrl[:], '###')
            print(physicsSimulation.data.qvel, '!!!')
            print(physicsSimulation.data.qpos, '~~~')
            print(physicsSimulation.data.body_xpos, '...')
        if frameIndex % 20 ==   0 and frameIndex > 200:
            action = np.array([7,7, -10, 0])
            physicsSimulation.data.ctrl[:] = action
        if frameIndex % 1 == 0 and frameIndex > 500:
            action = np.array([7, 7, -10,0])
            physicsSimulation.data.ctrl[:] = action

            physicsSimulation.data.ctrl[:] = action
        vels = physicsSimulation.data.qvel
        #maxVelInAllAgents = vels[2]
        # maxVelInAllAgents = max([np.linalg.norm(vels[i:i+3]) for i in range(3)])
        # if maxVelInAllAgents > totalMaxVel:
        #     totalMaxVel = maxVelInAllAgents
        physicsSimulation.step()
        physicsSimulation.forward()
        physicsViewer.render()

    print(totalMaxVel)


    # baselinePhysicsViewer = mujoco.MjViewer(baselinePhysicsSimulation)
    # numSimulationFrames = 0
    # for frameIndex in range(numSimulationFrames):
    #     #action = np.array([0] * 24)
    #     #baselinePhysicsSimulation.data.ctrl[:] = action
    #     baselinePhysicsSimulation.step()
    #     baselinePhysicsSimulation.forward()
    #     baselinePhysicsViewer.render()

if __name__ == '__main__':
    main()
