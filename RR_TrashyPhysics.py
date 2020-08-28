import pygame
from pygame.sprite import Sprite
from typing import List, Dict, Tuple
import RR_Constants as const
from RR_Robot import Robot
from RR_Ball import Ball
from RR_Goal import Goal

def robots_collided(sprRobot1: Robot, sprRobot2: Robot) -> bool:
    # TODO account for rotation here
    return pygame.sprite.collide_rect(sprRobot1, sprRobot2)

def ball_robot_collided(sprBall: Ball, sprRobot: Robot) -> bool:
    # TODO account for rotation here
    return pygame.sprite.collide_rect(sprRobot, sprBall)

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
        dblDeltaX = sprRobot.dblRect.centerx - sprRobot.dblRectPrior.centerx
        dblDeltaY = sprRobot.dblRect.centery - sprRobot.dblRectPrior.centery

        if dblDeltaX > 0:
            sprBall.dblXVelocity = max(dblDeltaX, sprBall.dblXVelocity)
        else:
            sprBall.dblXVelocity = min(dblDeltaX, sprBall.dblXVelocity)

        if dblDeltaY > 0:
            sprBall.dblYVelocity = max(dblDeltaY, sprBall.dblYVelocity)
        else:
            sprBall.dblYVelocity = min(dblDeltaY, sprBall.dblYVelocity)

        # TODO this is kind of trash but whatever
        sprBall.dblRect.left += dblDeltaX
        sprBall.dblRect.top += dblDeltaY
        sprBall.rect.left = sprBall.dblRect.left
        sprBall.rect.top = sprBall.dblRect.top

def bounce_balls(spr1: Ball, spr2: Ball) -> None:
    print("TrashyPhysics.bounce_balls() not done.")

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

def coord_within_robot(tplCoord: Tuple[float,float], sprRobot: Robot):
    dblX, dblY = tplCoord  # unpack

