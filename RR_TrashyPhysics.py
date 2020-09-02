import pygame
from pygame.sprite import Sprite
from typing import List, Dict, Tuple
import RR_Constants as const
from RR_Robot import Robot #TODO reenable
from RR_Ball import Ball
from RR_Goal import Goal
from MyUtils import FloatRect, RightTriangle, Point, Div0
import math

def robot_in_goal(sprRobot: Robot, sprGoal: Goal) -> bool:
    for tplCorner in sprRobot.rectDbl.corners:
        if sprGoal.triShape.contains_point(tplCorner):
            return True

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
_rectBallInner = FloatRect(
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
    _rectBallInner.center = sprBall.rectDbl.center
    if sprRobot.dblRotation % 90 == 0:
        # robot is pure rect.
        # "Tangent" points of the ball for 0,90,180,270 occur at 45 rot (or equiv)
        _rectBallInner.rotation = 45
        for tplPoint in _rectBallInner.corners:
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
        _rectBallInner.rotation = 45
        lstKeyPointsOnBall = _rectBallInner.corners

        # 3rd side will be rotated according to robot's rotation
        _rectBallInner.rotation = sprRobot.dblRotation
        lstKeyPointsOnBall += _rectBallInner.corners

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
        rectPrior = sprRobot.rectDblPriorFrame
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
        if sprBall.lngFrameMass < sprRobot.lngFrameMass:
            sprBall.rectDbl.left += dblDeltaX
            sprBall.rectDbl.top += dblDeltaY
            sprBall.rect.left = sprBall.rectDbl.left
            sprBall.rect.top = sprBall.rectDbl.top
            sprBall.lngFrameMass = sprRobot.lngFrameMass  # for the rest of the frame, consider them one entity
        else:
            """Ball hit the wall or another robot. Don't budge."""

def bounce_balls(spr1: Ball, spr2: Ball) -> None:
    if spr1.rectDbl.center == spr2.rectDbl.center:
        raise Exception("Really tho?? The balls are in the EXACT same spot????")

    # Determine the angle of the line between the two centers
    dblRotRadians = (2 * math.pi) - math.atan(Div0(spr2.rectDbl.centery - spr1.rectDbl.centery, spr2.rectDbl.centerx - spr1.rectDbl.centerx))
    # dblRotRadians = math.atan(Div0(spr2.rectDbl.centery - spr1.rectDbl.centery, spr2.rectDbl.centerx - spr1.rectDbl.centerx))
    dblDegrees = math.degrees(dblRotRadians)
    dblCos = math.cos(dblRotRadians)
    dblSin = math.sin(dblRotRadians)

    # Determine point on Ball 1 closest to the center of Ball 2
    tplPnt1 = Point(
        x=spr1.rectDbl.centerx + const.BALL_RADIUS * dblCos,
        y=spr1.rectDbl.centery - const.BALL_RADIUS * dblSin
    )
    # and vice versa
    tplPnt2 = Point(
        x=spr2.rectDbl.centerx - const.BALL_RADIUS * dblCos,
        y=spr2.rectDbl.centery + const.BALL_RADIUS * dblSin
    )

    dblHalfDeltaX = (tplPnt2.x - tplPnt1.x) / 2
    dblHalfDeltaY = (tplPnt2.y - tplPnt1.y) / 2
    if spr1.lngFrameMass == spr2.lngFrameMass:
        """ Move balls out of each other. """
        spr1.rectDbl.centerx += dblHalfDeltaX
        spr1.rectDbl.centery += dblHalfDeltaY
        spr2.rectDbl.centerx -= dblHalfDeltaX
        spr2.rectDbl.centery -= dblHalfDeltaY
    else:
        """ Only one ball moves. """
        if spr1.lngFrameMass > spr2.lngFrameMass:
            spr2.rectDbl.centerx += tplPnt2.x - tplPnt1.x
            spr2.rectDbl.centery += tplPnt2.y - tplPnt1.y
            spr2.lngFrameMass = spr1.lngFrameMass  # and now they are one
        else:
            spr1.rectDbl.centerx += tplPnt1.x - tplPnt2.x
            spr1.rectDbl.centery += tplPnt1.y - tplPnt2.y
            spr1.lngFrameMass = spr2.lngFrameMass  # and now they are one

    """ Bounce! """
    # Let's conserve some momentum with vector projection (ya i totally remembered how to do this)
    # projection of 1's velocity onto contact vector:
    tplVect1to2 = Point(x=spr2.rectDbl.centerx - spr1.rectDbl.centerx, y=spr2.rectDbl.centery - spr1.rectDbl.centery)
    tplVect2to1 = Point(x=tplVect1to2.x * -1, y=tplVect1to2.y * -1)
    dblCenterDist_Sq = tplVect1to2.x*tplVect1to2.x + tplVect1to2.y*tplVect1to2.y
    dblProjClcTerm1 = Div0(tplVect1to2.x * spr1.dblXVelocity + tplVect1to2.y * spr1.dblYVelocity, dblCenterDist_Sq)
    tplVel1 = Point(x=dblProjClcTerm1 * tplVect1to2.x,y=dblProjClcTerm1 * tplVect1to2.y)
    dblProjClcTerm2 = Div0(tplVect2to1.x * spr2.dblXVelocity + tplVect2to1.y * spr2.dblYVelocity, dblCenterDist_Sq)
    tplVel2 = Point(x=dblProjClcTerm2 * tplVect2to1.x, y=dblProjClcTerm2 * tplVect2to1.y)
    tplDiff = Point(x=tplVel1.x - tplVel2.x, y=tplVel1.y - tplVel2.y)
    spr1.dblXVelocity -= tplDiff.x * const.BOUNCE_K_BALL
    spr1.dblYVelocity -= tplDiff.y * const.BOUNCE_K_BALL
    spr2.dblXVelocity += tplDiff.x * const.BOUNCE_K_BALL
    spr2.dblYVelocity += tplDiff.y * const.BOUNCE_K_BALL
    # this is a totally different sort of trash....



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

