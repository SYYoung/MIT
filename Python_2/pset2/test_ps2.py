from ps2 import RectangularRoom
from ps2 import Position
from ps2 import Robot
import random

random.seed(0)
rec = RectangularRoom(3, 4)
testcase = 2
if testcase == 1:
    ans1 = rec.getNumTiles()
    rec.cleanTileAtPosition(Position(0.5, 3.5))
    rec.cleanTileAtPosition(Position(1.5, 2.5))
    rec.cleanTileAtPosition(Position(2.5, 1.5))
    ans1 = rec.getNumCleanedTiles()
    ans1 = rec.isTileCleaned(2,1)
    ans1 = rec.isTileCleaned(2,3)
    ans1 = rec.isPositionInRoom(Position(2.1, 1.9))
    ans1 = rec.isPositionInRoom(Position(3.1, 2.9))
    ans1 = rec.getRandomPosition()

if testcase == 2:
    speed = 0.8
    robot1 = Robot(rec, speed)
    ans1 = robot1.getRobotPosition()
    ans1 = robot1.getRobotDirection()