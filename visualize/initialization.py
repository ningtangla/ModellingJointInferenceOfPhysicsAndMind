import pygame as pg

def initializeScreen(fullScreen, screenWidth, screenHeight):
    pg.init()
    if fullScreen:
        screen = pg.display.set_mode((screenWidth, screenHeight), pg.FULLSCREEN)
    else:
        screen = pg.display.set_mode((screenWidth, screenHeight))
    return screen