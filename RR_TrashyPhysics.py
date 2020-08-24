import pygame
import RR_Constants as const

# def check_collision(spr1, spr2):
#     if isinstance(spr1, Ball):
#         if isinstance(spr2, Ball):
#         elif isinstance(spr2, Robot):
#         elif isinstance(spr2, Goal):
#         else:
#             raise Exception("Unknown object: " & type(spr2).__name__)
#     elif isinstance(spr1, Robot):
#     elif isinstance(spr1, Goal):
#     else:
#         raise Exception("Unknown object: " & type(spr1).__name__)
#     print("stuff")
#
# def resolve_collision(spr1, spr2):
#     print("stuff")
def __init__(self):
    raise Exception("Dont instantiate this. Actually, this should be in a separate file.")

def check_robot_robot_collision(sprRobot1, sprRobot2):
    # TODO account for rotation here
    return pygame.sprite.collide_rect(sprRobot1, sprRobot2)

def resolve_robot_robot_collision(sprRobot1, sprRobot2):
    raise NotImplementedError("jlskdj")

def check_ball_robot_collision(sprBall, sprRobot):
    # TODO account for rotation here
    return pygame.sprite.collide_rect(sprRobot, sprBall)

def resolve_ball_robot_collision(sprBall, sprRobot):
    raise NotImplementedError("jlskdj")

def check_collision_wall(objEntity):
    if isinstance(objEntity, pygame.Rect):
        rect = objEntity
    else:
        rect = objEntity.rect

    return rect.left < 0 or \
           rect.right > const.ARENA_WIDTH or\
           rect.top < 0 or\
           rect.bottom > const.ARENA_HEIGHT