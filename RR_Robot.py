import pygame
from typing import Tuple, Dict, List
import math
import random
import RR_Constants as const
from MyUtils import FloatRect

class Robot(pygame.sprite.Sprite):

    ssurfHappyRobot = pygame.image.load("Happy_Robot_40x20.png").convert()
    ssurfHappyRobot.set_colorkey((0,0,0), pygame.RLEACCEL)
    slngHappyRobotInitialRot = 90
    ssurfGrumpyRobot = pygame.image.load("Grumpy_Robot_40x20.png").convert()
    ssurfGrumpyRobot.set_colorkey((0, 0, 0), pygame.RLEACCEL)
    slngGrumpyRobotInitialRot = 90

    def __init__(self, intTeam):
        super(Robot, self).__init__()

        self.intTeam = intTeam
        self.lngLThrust = 0
        self.lngRThrust = 0
        self.lngMoveSpeed = const.ROBOT_VEL
        self.lngMoveSpeedRem = 0
        self._move_linear_remainder_x = 0.0
        self._move_linear_remainder_y = 0.0
        # self._move_angular_remainder_x = 0.0
        # self._move_angular_remainder_y = 0.0
        # This works better than the remainder approach (aka i fucked something up with the latter)
        self._move_angular_roundup_toggle = False

        if intTeam == const.TEAM_HAPPY:
            self.surfBase = Robot.ssurfHappyRobot
            self.surf = self.surfBase.copy()
            self.dblRotation = Robot.slngHappyRobotInitialRot
            self.dblRotationPrior = Robot.slngHappyRobotInitialRot
            self.dblInitialImageOffset = Robot.slngHappyRobotInitialRot
            self.rect = self.surf.get_rect(
                center=(  # Happy team starts in the bottom
                    random.randint(const.ROBOT_WIDTH/2, const.ARENA_WIDTH - const.ROBOT_WIDTH/2),
                    random.randint(const.ARENA_HEIGHT/2 + const.ROBOT_LENGTH/2, const.ARENA_HEIGHT - const.ROBOT_LENGTH/2)
                )
            )
        else:
            self.surfBase = Robot.ssurfGrumpyRobot
            self.surf = self.surfBase.copy()
            self.dblRotation = Robot.slngGrumpyRobotInitialRot
            self.dblRotationPrior = Robot.slngGrumpyRobotInitialRot
            self.dblInitialImageOffset = Robot.slngGrumpyRobotInitialRot
            self.rect = self.surf.get_rect(
                center=(  # Grumpy team starts in the top
                    random.randint(const.ROBOT_WIDTH/2, const.ARENA_WIDTH - const.ROBOT_WIDTH/2),
                    random.randint(const.ROBOT_LENGTH/2, const.ARENA_HEIGHT/2 - const.ROBOT_LENGTH/2)
                )
            )

        # self.rect = rendering (integers), self.dblRect = location calc (float)
        self.dblRect = FloatRect(self.rect.left, self.rect.right, self.rect.top, self.rect.bottom)
        self.dblRectPrior = self.dblRect.copy()

    def set_thrust(self, lngLThrust, lngRThrust):
        self.lngLThrust = lngLThrust
        self.lngRThrust = lngRThrust

    def on_step_begin(self):
        self.lngMoveSpeedRem = self.lngMoveSpeed
        self.dblRectPrior = self.dblRect.copy()
        self.dblRotationPrior = self.dblRotation

    def on_step_end(self):
        self.rect.center = self.dblRect.center

    def move_all(self):
        if self.lngMoveSpeedRem <= 0:
            return False

        self._move_internal(self.lngMoveSpeedRem)
        self.lngMoveSpeedRem = 0

    def move_one(self):
        if self.lngMoveSpeedRem <= 0:
            return False

        self._move_internal(1)
        self.lngMoveSpeedRem -= 1

    def undo_move(self):
        self.lngMoveSpeedRem = self.lngMoveSpeed
        self.dblRect = self.dblRectPrior.copy()

        if self.dblRotation != self.dblRotationPrior:
            self.dblRotation = self.dblRotationPrior
            self.surf = pygame.transform.rotate(self.surfBase, self.dblRotation - self.dblInitialImageOffset)
            self.rect = self.surf.get_rect(center=self.dblRect.center)
        else:
            self.rect.center = self.dblRect.center

    def _move_internal(self, lngSpeed):

        if self.lngLThrust == self.lngRThrust:  # Linear travel
            if self.lngLThrust < 0:
                self._move_linear(lngSpeed * -1)
            elif self.lngLThrust > 0:
                self._move_linear(lngSpeed)
            else:  # no thrust, just consume all movement speed
                self.lngMoveSpeedRem = 0
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
                    self.dblRect.centerx + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians),
                    self.dblRect.centery - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)
                )
                self._move_angular(lngAngularVel, tplCenterRot, -90)
            else:  # rotate around right track
                dblRadians = math.radians(self.dblRotation - 90)
                tplCenterRot = (  # center of right track
                    self.dblRect.centerx + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians),
                    self.dblRect.centery - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)
                )
                self._move_angular(lngAngularVel, tplCenterRot, 90)

    def _move_linear(self, lngVel):
        dblRotRadians = math.radians(self.dblRotation)
        self.dblRect.left += math.cos(dblRotRadians) * float(lngVel)
        self.dblRect.top += math.sin(dblRotRadians) * float(lngVel) * -1

        # If we drive into the wall at an angle, let's pretend it's fine to "slide" along
        # TODO probly don't do this though
        # TODO also make this smarter and calc properly based on rotation
        if self.dblRect.left < 0:
            self.dblRect.left = 0
        if self.dblRect.right > const.ARENA_WIDTH:
            self.dblRect.right = const.ARENA_WIDTH
        if self.dblRect.top <= 0:
            self.dblRect.top = 0
        if self.dblRect.bottom >= const.ARENA_HEIGHT:
            self.dblRect.bottom = const.ARENA_HEIGHT

        self.rect.center = self.dblRect.center

    def _move_angular(self, dblAngularVel:float, tplCenterRot:Tuple[float,float]=None, dblTrackToCenterAngleAdj:float=0):
        self.dblRotation += dblAngularVel
        self.dblRotation %= 360
        if self.dblRotation < 0:
            self.dblRotation += 360

        if tplCenterRot:  # We're not rotating in place. Adjust rect center accordingly.
            dblRadians = math.radians(self.dblRotation + dblTrackToCenterAngleAdj)
            self.dblRect.centerx = tplCenterRot[0] + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians)
            self.dblRect.centery = tplCenterRot[1] - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)

        self.surf = pygame.transform.rotate(self.surfBase, self.dblRotation - self.dblInitialImageOffset)
        self.rect = self.surf.get_rect(center=self.dblRect.center)
        # rect width/height has changed due to the rotation - reset on dblRect
        self.dblRect.width = self.rect.width
        self.dblRect.height = self.rect.height

        # If we rotate into the wall... fuck it, just pop ourselves out for now
        # TODO probly don't actually do this though
        if self.dblRect.left < 0:
            self.dblRect.left = 0
        if self.dblRect.right > const.ARENA_WIDTH:
            self.dblRect.right = const.ARENA_WIDTH
        if self.dblRect.top <= 0:
            self.dblRect.top = 0
        if self.dblRect.bottom >= const.ARENA_HEIGHT:
            self.dblRect.bottom = const.ARENA_HEIGHT

        self.rect.center = self.dblRect.center
