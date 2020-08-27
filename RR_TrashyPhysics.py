import pygame
import RR_Constants as const
from RR_Robot import Robot
from RR_Ball import Ball
from RR_Goal import Goal

def check_robot_robot_collision(sprRobot1: Robot, sprRobot2: Robot) -> bool:
    # TODO account for rotation here
    return pygame.sprite.collide_rect(sprRobot1, sprRobot2)

def resolve_robot_robot_collision(sprRobot1: Robot, sprRobot2: Robot) -> None:
    # not a super elegant way of doing this but... its the easiest!
    sprRobot1.undo_move()
    sprRobot2.undo_move()

    # Smash them again!
    lngSmashCount = 0
    while not check_robot_robot_collision(sprRobot1, sprRobot2):
        lngSmashCount += 1
        sprRobot1.move_one()
        sprRobot2.move_one()
        if sprRobot1.lngMoveSpeedRem == 0 or sprRobot2.lngMoveSpeedRem == 0:
            return  # theoretically possible due to rounding differences

    # Once more! but don't full smash!
    sprRobot1.undo_move()
    sprRobot2.undo_move()
    for _ in range(lngSmashCount):
        sprRobot1.move_one()
        sprRobot2.move_one()

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