import pygame
from . import GameEnv, GameEnv_Simple, Ball, Robot, Goal
from typing import Tuple, List, Dict
import random

class AbstractPlayer:
    def __init__(self, env: GameEnv, robot: Robot):
        self.env = env
        self.robot = robot

    def get_action(self) -> Tuple[float, float]:
        raise Exception("Override this in the child class.")

class OG_Twitchy(AbstractPlayer):
    def get_action(self) -> Tuple[float, float]:
        rando = random.random()
        # ~5% chance to turn left or right, 45% chance to go forward/back
        if rando <= 0.05:
            # turn left
            action = (-1, 1)
        elif rando <= 0.5:
            # go straight
            action = (1, 1)
        elif rando < 0.95:
            # go back
            action = (-1, -1)
        else:
            # turn right
            action = (1, -1)
        return action



class Human(AbstractPlayer):
    def __init__(self, env: GameEnv,
                 robot: Robot,
                 key_left=pygame.K_a,
                 key_right=pygame.K_d,
                 key_forwards=pygame.K_w,
                 key_backwards=pygame.K_s):

        super(Human, self).__init__(env, robot)
        self.key_left = key_left
        self.key_right = key_right
        self.key_forwards = key_forwards
        self.key_backwards = key_backwards

    def get_action(self) -> Tuple[float, float]:
        pygame.event.get()  # If you don't call this first, doesn't work... worth investigating at some point

        # Process player input
        dctKeyDown = pygame.key.get_pressed()
        lngLThrust = 0
        lngRThrust = 0
        if dctKeyDown[self.key_forwards]:
            lngLThrust += 1
            lngRThrust += 1
        if dctKeyDown[self.key_backwards]:
            lngLThrust -= 1
            lngRThrust -= 1
        if dctKeyDown[self.key_left]:
            lngLThrust -= 1
            lngRThrust += 1
        if dctKeyDown[self.key_right]:
            lngLThrust += 1
            lngRThrust -= 1

        return (lngLThrust, lngRThrust)


class DistantHuman(Human):
    def __init__(self, env: GameEnv, robot: Robot):
        super(Human, self).__init__(env, robot)
        raise NotImplementedError("SOMEBODY SHOULD TOTALLY MAKE A CLIENT/SERVER PLAYER THO")



