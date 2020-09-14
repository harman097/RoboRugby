import pygame
from typing import Tuple, List
import math
import robo_rugby.gym_env.RR_Constants as const
from MyUtils import FloatRect

class Robot(pygame.sprite.Sprite):

    __instance_count = 0

    ssurfHappyRobot = pygame.image.load("robo_rugby/resources/Happy_Robot_40x20.png").convert()
    ssurfHappyRobot.set_colorkey((0,0,0), pygame.RLEACCEL)
    slngHappyRobotInitialRot = 90
    ssurfGrumpyRobot = pygame.image.load("robo_rugby/resources/Grumpy_Robot_40x20.png").convert()
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


    def __init__(self, intTeam:int, tplCenter:Tuple[float,float]):
        super(Robot, self).__init__()
        Robot.__instance_count += 1
        self.__id = Robot.__instance_count

        self.bln_allow_wall_sliding = False
        self.intTeam = intTeam
        self.lngLThrust = 0
        self.lngRThrust = 0
        self.lngFrameMass = const.MASS_ROBOT

        # self.rect = rendering (integers), self.dblRect = location calc (float)
        self.rectDbl = FloatRect(0, const.ROBOT_LENGTH, 0, const.ROBOT_WIDTH)
        self.rectDbl.center = tplCenter

        if intTeam == const.TEAM_HAPPY:
            self.surfBase = Robot.ssurfHappyRobot
            self.rectDbl.rotation = Robot.slngHappyRobotInitialRot
        else:
            self.surfBase = Robot.ssurfGrumpyRobot
            self.rectDbl.rotation = Robot.slngGrumpyRobotInitialRot

        self.rectDblPriorStep = self.rectDbl.copy()
        self.lngMoveCount = 0
        self._lstStates = [None]*self._slngMoveHistorySize # type: List[Tuple[float, float, float, int]]

    def on_reset(self):
        self.__init__(self.intTeam, self.rectDbl.center)

    def _store_state(self):
        self._lstStates[self.lngMoveCount % self._slngMoveHistorySize] = (
            self.rectDbl.centerx,
            self.rectDbl.centery,
            self.dblRotation,
            self.lngMoveCount
        )

    # todo once we get the robot, re-evaluate this
    # Are the robot engines just "on" or "off"? or can you specify how much power?
    def set_thrust(self, l_thrust: float, r_thrust: float):
        self.lngLThrust = int(round(l_thrust))
        self.lngRThrust = int(round(r_thrust))

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

    def _move_linear(self, lngVel:int):
        dblRotRadians = math.radians(self.dblRotation)
        tplCenterPrior = self.rectDbl.center
        self.rectDbl.left += math.cos(dblRotRadians) * float(lngVel)
        self.rectDbl.top += math.sin(dblRotRadians) * float(lngVel) * -1

        bln_hit_wall = False
        if self.rectDbl.left < 0:
            self.rectDbl.left = 0
            bln_hit_wall = True
        if self.rectDbl.right > const.ARENA_WIDTH:
            self.rectDbl.right = const.ARENA_WIDTH
            bln_hit_wall = True
        if self.rectDbl.top <= 0:
            self.rectDbl.top = 0
            bln_hit_wall = True
        if self.rectDbl.bottom >= const.ARENA_HEIGHT:
            self.rectDbl.bottom = const.ARENA_HEIGHT
            bln_hit_wall = True

        if bln_hit_wall and not self.bln_allow_wall_sliding:
            self.rectDbl.center = tplCenterPrior

    def _move_angular(self, dblAngularVel:float,
                      tplCenterRot:Tuple[float,float] = None,
                      dblTrackToCenterAngleAdj:float = 0):
        dblRotationPrior = dblAngularVel
        tplCenterPrior = self.rectDbl.center
        self.dblRotation += dblAngularVel

        if tplCenterRot:  # We're not rotating in place. Adjust rect center accordingly.
            dblRadians = math.radians(self.dblRotation + dblTrackToCenterAngleAdj)
            self.rectDbl.centerx = tplCenterRot[0] + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians)
            self.rectDbl.centery = tplCenterRot[1] - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)

        bln_hit_wall = False
        if self.rectDbl.left < 0:
            self.rectDbl.left = 0
            bln_hit_wall = True
        if self.rectDbl.right > const.ARENA_WIDTH:
            self.rectDbl.right = const.ARENA_WIDTH
            bln_hit_wall = True
        if self.rectDbl.top <= 0:
            self.rectDbl.top = 0
            bln_hit_wall = True
        if self.rectDbl.bottom >= const.ARENA_HEIGHT:
            self.rectDbl.bottom = const.ARENA_HEIGHT
            bln_hit_wall = True

        if bln_hit_wall and not self.bln_allow_wall_sliding:
            self.rectDbl.center = tplCenterPrior
            self.rectDbl.rotation = dblRotationPrior



    #endregion

    def __str__(self):
        return f"Robot {self.__instance_count} {self.rectDbl.center}"
    def __repr__(self):
        return f"Robot {self.__instance_count} {self.rectDbl.center}"