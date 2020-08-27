import pygame
import RR_Constants as const
from RR_Robot import Robot
from RR_Ball import Ball
from RR_Goal import Goal

def robots_collided(sprRobot1: Robot, sprRobot2: Robot) -> bool:
    # TODO account for rotation here
    return pygame.sprite.collide_rect(sprRobot1, sprRobot2)

def resolve_robot_robot_collision(sprRobot1: Robot, sprRobot2: Robot) -> None:
    # not a super elegant way of doing this but... its the easiest!
    sprRobot1.undo_move()
    sprRobot2.undo_move()

    # Smash them again!
    lngNoSmashCount = 0
    while sprRobot1.lngMoveSpeedRem > 0 and sprRobot2.lngMoveSpeedRem > 0:
        sprRobot1.move_one()
        sprRobot2.move_one()
        if not robots_collided(sprRobot1, sprRobot2):
            lngNoSmashCount += 1
        else:
            # Once more! but don't full smash!
            sprRobot1.undo_move()
            sprRobot2.undo_move()
            for _ in range(lngNoSmashCount):
                sprRobot1.move_one()
                sprRobot2.move_one()
            return

def check_ball_robot_collision(sprBall: Ball, sprRobot: Robot) -> bool:
    # TODO account for rotation here
    return pygame.sprite.collide_rect(sprRobot, sprBall)

def resolve_ball_robot_collision(sprBall: Ball, sprRobot: Robot) -> None:
    raise NotImplementedError("jlskdj")

def check_collision_wall(objEntity) -> bool:
    if isinstance(objEntity, pygame.Rect):
        rect = objEntity
    else:
        rect = objEntity.rect

    return rect.left < 0 or \
           rect.right > const.ARENA_WIDTH or\
           rect.top < 0 or\
           rect.bottom > const.ARENA_HEIGHT