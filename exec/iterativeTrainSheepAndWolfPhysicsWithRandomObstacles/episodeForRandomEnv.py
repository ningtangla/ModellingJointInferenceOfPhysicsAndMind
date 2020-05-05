
class SampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction
        self.render = render
        self.renderOn = renderOn

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            if self.renderOn:
                self.render(state, runningStep)
            actionDists = policy(state)
            action = [choose(action) for choose, action in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class RenderInObstacle():
    def __init__(self, numOfAgent, posIndex, screen, circleColorList,saveImage, saveImageDir,scaleState,drawState):
        self.numOfAgent = numOfAgent
        self.posIndex = posIndex
        self.screen = screen
        self.circleColorList = circleColorList
        self.saveImage  = saveImage
        self.saveImageDir = saveImageDir
        self.scaleState=scaleState
        self.drawState=drawState
    def __call__(self, state, timeStep):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()

            rescaleState=self.scaleState(state)
            # print(state,rescaleState)
            screen = self.drawState(self.numOfAgent,rescaleState)
            pg.time.wait(100)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(self.screen, self.saveImageDir + '/' + format(timeStep, '05') + ".png")
class DrawState:
    def __init__(self, screen, circleSizeList,circleColorList, positionIndex, drawBackGround):
        self.screen = screen
        self.circleSizeList = circleSizeList
        self.xIndex, self.yIndex = positionIndex
        self.drawBackGround = drawBackGround
        self.circleColorList=circleColorList
    def __call__(self, numOfAgent, state):

        self.drawBackGround()

        for agentIndex in range(numOfAgent):
            agentPos =[np.int(pos) for pos in state[agentIndex]]
            agentColor = self.circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSizeList[agentIndex])
        pg.display.flip()
        return self.screen

class ScaleState:
    def __init__(self, positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange):
        self.xIndex, self.yIndex = positionIndex
        self.rawXMin, self.rawXMax = rawXRange
        self.rawYMin, self.rawYMax = rawYRange

        self.scaledXMin, self.scaledXMax = scaledXRange
        self.scaledYMin, self.scaledYMax = scaledYRange

    def __call__(self, originalState):
        xScale = (self.scaledXMax - self.scaledXMin) / (self.rawXMax - self.rawXMin)
        yScale = (self.scaledYMax - self.scaledYMin) / (self.rawYMax - self.rawYMin)

        adjustX = lambda rawX: (rawX - self.rawXMin) * xScale + self.scaledXMin
        adjustY = lambda rawY: (self.rawYMax-rawY) * yScale + self.scaledYMin

        adjustState = lambda state: [adjustX(state[self.xIndex]), adjustY(state[self.yIndex])]

        newState = [adjustState(agentState) for agentState in originalState]

        return newState

class TransferWallToRescalePosForDraw:
    def __init__(self,rawXRange,rawYRange,scaledXRange,scaledYRange):
        self.rawXMin, self.rawXMax = rawXRange
        self.rawYMin, self.rawYMax = rawYRange
        self.scaledXMin, self.scaledXMax = scaledXRange
        self.scaledYMin, self.scaledYMax = scaledYRange
        xScale = (self.scaledXMax - self.scaledXMin) / (self.rawXMax - self.rawXMin)
        yScale = (self.scaledYMax - self.scaledYMin) / (self.rawYMax - self.rawYMin)
        adjustX = lambda rawX: (rawX - self.rawXMin) * xScale + self.scaledXMin
        adjustY = lambda rawY: (self.rawYMax-rawY) * yScale + self.scaledYMin
        self.rescaleWall=lambda wallForDraw :[adjustX(wallForDraw[0]),adjustY(wallForDraw[1]),wallForDraw[2]*xScale,wallForDraw[3]*yScale]
        self.tranferWallForDraw=lambda wall:[wall[0]-wall[2],wall[1]+wall[3],2*wall[2],2*wall[3]]
    def __call__(self,wallList):

        wallForDarwList=[self.tranferWallForDraw(wall) for wall in wallList]
        allObstaclePos=[ self.rescaleWall(wallForDraw) for wallForDraw in wallForDarwList]
        return allObstaclePos

class DrawBackgroundWithObstacles:
    def __init__(self, screen, screenColor, xBoundary, yBoundary, allObstaclePos, lineColor, lineWidth):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.allObstaclePos = allObstaclePos
        self.lineColor = lineColor
        self.lineWidth = lineWidth

    def __call__(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    exit()
        self.screen.fill(self.screenColor)
        rectPos = [self.xBoundary[0], self.yBoundary[0], self.xBoundary[1], self.yBoundary[1]]
        pg.draw.rect(self.screen, self.lineColor, rectPos, self.lineWidth)
        [pg.draw.rect(self.screen, self.lineColor, obstaclePos, self.lineWidth) for obstaclePos in
         self.allObstaclePos]

        return