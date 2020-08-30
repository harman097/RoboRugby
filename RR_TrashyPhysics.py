import pygame
from pygame.sprite import Sprite
from typing import List, Dict, Tuple
import RR_Constants as const
from RR_Robot import Robot
from RR_Ball import Ball
from RR_Goal import Goal
from MyUtils import FloatRect, RightTriangle, Point
import math

def robots_collided(sprRobot1: Robot, sprRobot2: Robot) -> bool:
    if sprRobot1.rectDbl.contains_point(sprRobot2.rectDbl.center):
        return True

    if sprRobot2.rectDbl.contains_point(sprRobot1.rectDbl.center):
        return True

    for tplCorner in sprRobot1.rectDbl.corners:
        if sprRobot2.rectDbl.contains_point(tplCorner):
            return True

    for tplCorner in sprRobot2.rectDbl.corners:
        if sprRobot1.rectDbl.contains_point(tplCorner):
            return True

    return False

# square whose corners are on the surface of the ball.
# used to find different points on a ball's surface for a given rotation.
_dblHalfRadius_Rad2 = const.BALL_RADIUS * math.pow(2, .5) / 2
_rectInner = FloatRect(
    dblLeft=-1 * _dblHalfRadius_Rad2,
    dblRight=_dblHalfRadius_Rad2,
    dblTop=-1 * _dblHalfRadius_Rad2,
    dblBottom=_dblHalfRadius_Rad2
)
_dblBallRadius_Sq = const.BALL_RADIUS * const.BALL_RADIUS

def ball_robot_collided(sprBall: Ball, sprRobot: Robot) -> bool:
    if not pygame.sprite.collide_rect(sprRobot, sprBall):
        return False

    """
    Either one of the corners of the robot is inside the ball -OR-
    One of the points on the ball whose tangent is parallel to an edge
    of the robot is inside the robot.
    """
    _rectInner.center = sprBall.rectDbl.center
    if sprRobot.dblRotation % 90 == 0:
        # robot is pure rect.
        # "Tangent" points of the ball for 0,90,180,270 occur at 45 rot (or equiv)
        _rectInner.rotation = 45
        for tplPoint in _rectInner.corners:
            if sprRobot.rectDbl.contains_point(tplPoint):
                return True

        # Ball could have hit a corner
        for tplPoint in sprRobot.rectDbl.corners:
            dblDistCenter = math.pow(tplPoint.x - sprBall.rectDbl.centerx, 2)
            dblDistCenter += math.pow(tplPoint.y - sprBall.rectDbl.centery, 2)
            if dblDistCenter <= _dblBallRadius_Sq:  # within the radius = winn
                return True
    else:
        # robot is tilted. check each edge (as a right triangle)
        # 2 sides of the triangle will be at 0,90,180, or 270
        _rectInner.rotation = 45
        lstKeyPointsOnBall = _rectInner.corners

        # 3rd side will be rotated according to robot's rotation
        _rectInner.rotation = sprRobot.dblRotation
        lstKeyPointsOnBall += _rectInner.corners

        for shpTriangle in sprRobot.rectDbl.sides_as_right_triangles():
            for tplPoint in lstKeyPointsOnBall:
                if shpTriangle.contains_point(tplPoint):
                    return True

            # Ball could have hit a corner
            for tplPoint in [shpTriangle.tplHyp0, shpTriangle.tplHyp1]:
                dblDistCenter = math.pow(tplPoint.x - sprBall.rectDbl.centerx, 2)
                dblDistCenter += math.pow(tplPoint.y - sprBall.rectDbl.centery, 2)
                if dblDistCenter <= _dblBallRadius_Sq:  # within the radius = winn
                    return True

    return False


def balls_collided(spr1: Ball, spr2: Ball):
    return pygame.sprite.collide_circle(spr1, spr2)

def collided_wall(objEntity) -> bool:
    if isinstance(objEntity, pygame.Rect):
        rect = objEntity
    else:
        rect = objEntity.rect

    return rect.left < 0 or \
           rect.right > const.ARENA_WIDTH or\
           rect.top < 0 or\
           rect.bottom > const.ARENA_HEIGHT

def apply_force_to_ball(sprRobot: Robot, sprBall: Ball) -> None:

        # Trashy trash solely so it's interesting, for the moment
        rectPrior = sprRobot.rectDblPrior
        dblDeltaX = (sprRobot.rectDbl.centerx - rectPrior.centerx) * const.PUSH_FACTOR
        dblDeltaY = (sprRobot.rectDbl.centery - rectPrior.centery) * const.PUSH_FACTOR

        # TODO this part except not total trash
        # dblDeltaRot = sprRobot.rectDbl.rotation - rectPrior.rotation
        # if dblDeltaRot != 0:
        #     triFront = sprRobot.rectDbl.side_as_right_triangle(FloatRect.SideType.RIGHT)
        #     triBack = sprRobot.rectDbl.side_as_right_triangle(FloatRect.SideType.LEFT)
        #     _rectInner.center = sprBall.rectDbl.center
        #     _rectInner.rotation = sprBall.rectDbl.rotation
        #     for tplTanPoint in _rectInner.corners:
        #         if triFront.contains_point(tplTanPoint):
        #             if dblDeltaRot > 0:
        #                 tplStart = rectPrior.corner(FloatRect.CornerType.BOTTOM_RIGHT)
        #                 tplEnd = sprRobot.rectDbl.corner(FloatRect.CornerType.BOTTOM_RIGHT)
        #             else:
        #                 tplStart = rectPrior.corner(FloatRect.CornerType.TOP_RIGHT)
        #                 tplEnd = sprRobot.rectDbl.corner(FloatRect.CornerType.TOP_RIGHT)
        #             dblDeltaX += tplEnd.x - tplStart.x
        #             dblDeltaY += tplEnd.y - tplStart.y
        #             break
        #         elif triBack.contains_point(tplTanPoint):
        #             if dblDeltaRot > 0:
        #                 tplStart = rectPrior.corner(FloatRect.CornerType.TOP_LEFT)
        #                 tplEnd = sprRobot.rectDbl.corner(FloatRect.CornerType.TOP_LEFT)
        #             else:
        #                 tplStart = rectPrior.corner(FloatRect.CornerType.BOTTOM_LEFT)
        #                 tplEnd = sprRobot.rectDbl.corner(FloatRect.CornerType.BOTTOM_LEFT)
        #             dblDeltaX += tplEnd.x - tplStart.x
        #             dblDeltaY += tplEnd.y - tplStart.y
        #             break

        if dblDeltaX > 0:
            sprBall.dblXVelocity = max(dblDeltaX, sprBall.dblXVelocity)
        else:
            sprBall.dblXVelocity = min(dblDeltaX, sprBall.dblXVelocity)

        if dblDeltaY > 0:
            sprBall.dblYVelocity = max(dblDeltaY, sprBall.dblYVelocity)
        else:
            sprBall.dblYVelocity = min(dblDeltaY, sprBall.dblYVelocity)

        # TODO this is kind of trash but whatever
        sprBall.rectDbl.left += dblDeltaX
        sprBall.rectDbl.top += dblDeltaY
        sprBall.rect.left = sprBall.rectDbl.left
        sprBall.rect.top = sprBall.rectDbl.top

def bounce_balls(spr1: Ball, spr2: Ball) -> None:
    AvoidConsoleSpam = 'Yes'
    #print("TrashyPhysics.bounce_balls() not done.")

def collision_pairs_self(grpSprites: pygame.sprite.Group,
                         fncCollided=pygame.sprite.collide_rect) -> List[Tuple[Sprite,Sprite]]:
    lstPairs = []
    lstSpr = grpSprites.sprites()
    for i in range(len(lstSpr)):
        for j in range(i+1, len(lstSpr)):
            if fncCollided(lstSpr[i], lstSpr[j]):
                lstPairs.append((lstSpr[i], lstSpr[j]))
    return lstPairs

def collision_pairs(grpSprites1: pygame.sprite.Group,
                    grpSprites2: pygame.sprite.Group,
                    fncCollided=pygame.sprite.collide_rect) -> List[Tuple[Sprite,Sprite]]:
    lstPairs = []
    lstSpr1 = grpSprites1.sprites()
    lstSpr2 = grpSprites2.sprites()
    for i in range(len(lstSpr1)):
        for j in range(len(lstSpr2)):
            if fncCollided(lstSpr1[i], lstSpr2[j]):
                lstPairs.append((lstSpr1[i], lstSpr2[j]))
    return lstPairs

