import pygame
from typing import Tuple, Dict, List
import math
import random
from enum import Enum
import RR_Constants as const
from MyUtils import FloatRect, Point

class Robot(pygame.sprite.Sprite):

    ssurfHappyRobot = pygame.image.load("Happy_Robot_40x20.png").convert()
    ssurfHappyRobot.set_colorkey((0,0,0), pygame.RLEACCEL)
    slngHappyRobotInitialRot = 90
    ssurfGrumpyRobot = pygame.image.load("Grumpy_Robot_40x20.png").convert()
    ssurfGrumpyRobot.set_colorkey((0, 0, 0), pygame.RLEACCEL)
    slngGrumpyRobotInitialRot = -90

    # Keep history for: 1s * fps * moves/frame
    _slngMoveHistorySize = 1 * const.FRAMERATE * const.MOVES_PER_FRAME

    @property
    def dblRotation(self) -> float:
        return self.rectDbl.rotation

    @dblRotation.setter
    def dblRotation(self, dblNewRotation: float):
        self.rectDbl.rotation = dblNewRotation

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.rectDbl.left,
            self.rectDbl.top,
            self.rectDbl.right - self.rectDbl.left,
            self.rectDbl.bottom - self.rectDbl.top
        )

    @property
    def surf(self):
        return pygame.transform.rotate(self.surfBase, self.dblRotation)
        # self.rect = self.surf.get_rect(center=(self.rectDbl.center))

    @property
    def rectDblPriorFrame(self) -> FloatRect:
        lngI = (self.lngMoveCount - 1) % Robot._slngMoveHistorySize
        if self._lstStates[lngI] is None:
            return self.rectDbl.copy()

        rectPrior = self.rectDbl.copy()
        rectPrior.centerx, rectPrior.centery, rectPrior.rotation, lngMC = self._lstStates[lngI]
        if lngMC != self.lngMoveCount - 1:
            raise Exception(f"How...? {lngMC} != {self.lngMoveCount - 1}")

        return rectPrior


    def __init__(self, intTeam):
        super(Robot, self).__init__()

        self.intTeam = intTeam
        self.lngLThrust = 0
        self.lngRThrust = 0
        self.lngFrameMass = const.MASS_ROBOT

        # self.rect = rendering (integers), self.dblRect = location calc (float)
        self.rectDbl = FloatRect(0, const.ROBOT_LENGTH, 0, const.ROBOT_WIDTH)

        if intTeam == const.TEAM_HAPPY:
            self.surfBase = Robot.ssurfHappyRobot
            # Happy team starts in the bottom-right quad with 2-robots padding
            self.rectDbl.centerx = random.randint(const.ARENA_WIDTH / 2 + const.ROBOT_WIDTH * 2, const.ARENA_WIDTH - const.ROBOT_WIDTH * 2)
            self.rectDbl.centery = random.randint(const.ARENA_HEIGHT / 2 + const.ROBOT_LENGTH * 2, const.ARENA_HEIGHT - const.ROBOT_LENGTH * 2)
            self.rectDbl.rotation = Robot.slngHappyRobotInitialRot
        else:
            self.surfBase = Robot.ssurfGrumpyRobot
            # Grumpy team starts in the top-left quad with 2-robots padding
            self.rectDbl.centerx = random.randint(const.ROBOT_WIDTH * 2, const.ARENA_WIDTH / 2 - const.ROBOT_WIDTH * 2)
            self.rectDbl.centery = random.randint(const.ROBOT_LENGTH * 2, const.ARENA_HEIGHT / 2 - const.ROBOT_LENGTH * 2)
            self.rectDbl.rotation = Robot.slngGrumpyRobotInitialRot

        self.rectDblPriorStep = self.rectDbl.copy()
        self.lngMoveCount = 0
        self._lstStates = [None]*self._slngMoveHistorySize # type: List[Tuple[float, float, float, int]]

    def _store_state(self):
        self._lstStates[self.lngMoveCount % self._slngMoveHistorySize] = (
            self.rectDbl.centerx,
            self.rectDbl.centery,
            self.dblRotation,
            self.lngMoveCount
        )

    def set_thrust(self, lngLThrust, lngRThrust):
        self.lngLThrust = lngLThrust
        self.lngRThrust = lngRThrust

    #region Movement

    def move(self):
        self.lngMoveCount += 1
        self._move_internal(1)
        self._store_state()

    def undo_move(self) -> bool:
        if self._try_restore_state(self.lngMoveCount - 1):
            self.lngMoveCount -= 1
            return True
        return False

    def on_step_begin(self):
        self.rectDblPriorStep = self.rectDbl.copy()

    def _try_restore_state(self, lngMoveCount: int) -> bool:
        intI = lngMoveCount % self._slngMoveHistorySize
        if not self._lstStates[intI]:
            return False


        # unpack tuple
        dblCx, dblCy, dblRot, lngMC = self._lstStates[intI]
        if lngMC != lngMoveCount:  # state was overwritten
            return False

        self.rectDbl.centerx = dblCx
        self.rectDbl.centery = dblCy
        if dblRot != self.rectDbl.rotation:
            self.rectDbl.rotation = dblRot

        return True

    def _move_internal(self, lngSpeed):

        if self.lngLThrust == self.lngRThrust:  # Linear travel
            if self.lngLThrust < 0:
                self._move_linear(lngSpeed * -1)
            elif self.lngLThrust > 0:
                self._move_linear(lngSpeed)
            else:  # no thrust
                return

        elif self.lngLThrust + self.lngRThrust == 0:  # Spin in-place
            if self.lngRThrust > 0:  # positive rotation
                lngAngularVel = const.ROBOT_ANGULAR_VEL_BOTH * lngSpeed
            else:  # negative rotation
                lngAngularVel = -1 * const.ROBOT_ANGULAR_VEL_BOTH * lngSpeed

            self._move_angular(lngAngularVel)

        else:  # Rotate around a track

            # Determine angular velocity
            if self.lngRThrust > 0 or self.lngLThrust < 0:  # "positive" (ccw) rotation
                lngAngularVel = const.ROBOT_ANGULAR_VEL_ONE * lngSpeed
            else: # lngRThrust < 0 or lngLThrust > 0 -> "negative" (cw) rotation
                lngAngularVel = const.ROBOT_ANGULAR_VEL_ONE * lngSpeed * -1

            # Determine center of rotation
            if self.lngRThrust != 0:  # rotate around left track
                dblRadians = math.radians(self.dblRotation + 90)
                tplCenterRot = (  # center of left track
                    self.rectDbl.centerx + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians),
                    self.rectDbl.centery - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)
                )
                self._move_angular(lngAngularVel, tplCenterRot, -90)
            else:  # rotate around right track
                dblRadians = math.radians(self.dblRotation - 90)
                tplCenterRot = (  # center of right track
                    self.rectDbl.centerx + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians),
                    self.rectDbl.centery - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)
                )
                self._move_angular(lngAngularVel, tplCenterRot, 90)

    def _move_linear(self, lngVel):
        dblRotRadians = math.radians(self.dblRotation)
        self.rectDbl.left += math.cos(dblRotRadians) * float(lngVel)
        self.rectDbl.top += math.sin(dblRotRadians) * float(lngVel) * -1

        # If we drive into the wall at an angle, let's pretend it's fine to "slide" along
        if self.rectDbl.left < 0:
            self.rectDbl.left = 0
        if self.rectDbl.right > const.ARENA_WIDTH:
            self.rectDbl.right = const.ARENA_WIDTH
        if self.rectDbl.top <= 0:
            self.rectDbl.top = 0
        if self.rectDbl.bottom >= const.ARENA_HEIGHT:
            self.rectDbl.bottom = const.ARENA_HEIGHT

    def _move_angular(self, dblAngularVel:float, tplCenterRot:Tuple[float,float]=None, dblTrackToCenterAngleAdj:float=0):
        self.dblRotation += dblAngularVel

        if tplCenterRot:  # We're not rotating in place. Adjust rect center accordingly.
            dblRadians = math.radians(self.dblRotation + dblTrackToCenterAngleAdj)
            self.rectDbl.centerx = tplCenterRot[0] + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians)
            self.rectDbl.centery = tplCenterRot[1] - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)

        # If we rotate into the wall... fuck it, just pop ourselves out for now
        # TODO probly don't actually do this though
        if self.rectDbl.left < 0:
            self.rectDbl.left = 0
        if self.rectDbl.right > const.ARENA_WIDTH:
            self.rectDbl.right = const.ARENA_WIDTH
        if self.rectDbl.top <= 0:
            self.rectDbl.top = 0
        if self.rectDbl.bottom >= const.ARENA_HEIGHT:
            self.rectDbl.bottom = const.ARENA_HEIGHT

    #endregion
