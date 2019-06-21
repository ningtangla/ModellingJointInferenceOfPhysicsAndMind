import os
import numpy as np
import pygame as pg

sheepID = 1
numAgent = 2
numPosEachAgent = 2
positionIndex = 0
screenColor = [255, 255, 255]
circleColorList = [[50, 50, 50], [50, 255, 50]]
circleSize = 8
titleSize = 10


class Render:
	def __init__(self, screen, imagePath=None, delay=500):
		self.screen = screen
		self.imagePath = imagePath
		self.delay = delay

	def __call__(self, state, value=None, round=None):
		width, height = pg.display.get_surface().get_size()
		print(width)
		pg.font.init()
		for j in range(1):
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
			self.screen.fill(screenColor)
			for i in range(numAgent):
				oneAgentState = state[i]
				oneAgentPosition = np.array(oneAgentState[positionIndex: numPosEachAgent])
				pg.draw.circle(self.screen, circleColorList[i], [np.int(oneAgentPosition[0]), np.int(oneAgentPosition[1])], circleSize)
			if value is not None:
				sheepAgentPos = state[sheepID][positionIndex: numPosEachAgent]
				font = pg.font.Font(pg.font.get_default_font(), circleSize)
				text = font.render(str(value), True, (0, 0, 0))
				self.screen.blit(text, (sheepAgentPos[0]-circleSize/2, sheepAgentPos[1]-circleSize/2))
			if round is not None:
				font = pg.font.Font(pg.font.get_default_font(), titleSize)
				text = font.render("Round {}".format(round), True, (0, 0, 0))
				self.screen.blit(text, (width/2-titleSize*2, height - titleSize))
			pg.display.flip()
			if self.imagePath is not None:
				filenameList = os.listdir(self.imagePath)
				pg.image.save(self.screen, self.imagePath+'/'+str(len(filenameList))+'.png')
			pg.time.wait(self.delay)


if __name__ == '__main__':
	pass