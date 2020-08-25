import pygame
import math
import random
import RR_Constants as const

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
        self.rectPrior = self.rect.copy()

    def set_thrust(self, lngLThrust, lngRThrust):
        self.lngLThrust = lngLThrust
        self.lngRThrust = lngRThrust

    def on_step_begin(self):
        self.lngMoveSpeedRem = self.lngMoveSpeed
        self.rectPrior = self.rect.copy()
        self.dblRotationPrior = self.dblRotation

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

            self._move_angular(lngAngularVel, self.rect.center, 0)

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
                    self.rect.centerx + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians),
                    self.rect.centery - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)
                )

                # TODO remove or set up tests or something
                switch_ValidateCenterRot = {
                    0:lambda tplCB, tplCT: tplCB[0] == round(tplCT[0]) and \
                                           tplCB[1] - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER == round(tplCT[1]),

                    180: lambda tplCB, tplCT: tplCB[0] == round(tplCT[0]) and \
                                              tplCB[1] + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER == round(tplCT[1]),

                    90: lambda tplCB, tplCT: tplCB[1] == round(tplCT[1]) and \
                                              tplCB[0] - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER == round(tplCT[0]),

                    270: lambda tplCB, tplCT: tplCB[1] == round(tplCT[1]) and \
                                              tplCB[0] + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER == round(tplCT[0]),

                    45:  lambda tplCB, tplCT: abs((tplCB[0] - tplCT[0]) - (tplCB[1] - tplCT[1])) < 2,
                    135: lambda tplCB, tplCT: abs((tplCB[0] - tplCT[0]) + (tplCB[1] - tplCT[1])) < 2,
                    225: lambda tplCB, tplCT: abs((tplCB[0] - tplCT[0]) - (tplCB[1] - tplCT[1])) < 2,
                    315: lambda tplCB, tplCT: abs((tplCB[0] - tplCT[0]) + (tplCB[1] - tplCT[1])) < 2

                }

                if self.dblRotation in switch_ValidateCenterRot and \
                        not switch_ValidateCenterRot[self.dblRotation](self.rect.center, tplCenterRot):
                    raise Exception(f"Center of rotation calc is off bro.\n"
                                    f"Current rotation: {self.dblRotation}\n"
                                    f"Current center: {self.rect.center}\n"
                                    f"Calc'd center of rotation: {tplCenterRot}")

                self._move_angular(lngAngularVel, tplCenterRot, -90)
            else:  # rotate around right track
                dblRadians = math.radians(self.dblRotation - 90)
                tplCenterRot = (  # center of right track
                    self.rect.centerx + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians),
                    self.rect.centery - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)
                )
                self._move_angular(lngAngularVel, tplCenterRot, 90)

    def _move_linear(self, lngVel):
        dblRotRadians = math.radians(self.dblRotation)
        dblX = math.cos(dblRotRadians) * float(lngVel)
        dblY = math.sin(dblRotRadians) * float(lngVel) * -1
        lngX = int(dblX)
        lngY = int(dblY)
        self._move_linear_remainder_x += dblX - lngX
        self._move_linear_remainder_y += dblY - lngY

        # if we constantly round down, some smaller angles won't ever actually affect the movement
        if self._move_linear_remainder_x > 1:
            self._move_linear_remainder_x -= 1.0
            lngX += 1
        elif self._move_linear_remainder_x < -1:
            self._move_linear_remainder_x += 1.0
            lngX -= 1

        if self._move_linear_remainder_y > 1:
            self._move_linear_remainder_y -= 1.0
            lngY += 1
        elif self._move_linear_remainder_y < -1:
            self._move_linear_remainder_y += 1.0
            lngY -= 1

        self.rect.move_ip(lngX, lngY)

        # If we drive into the wall at an angle, let's pretend it's fine to "slide" along
        # TODO probly don't do this though
        # TODO also make this smarter and calc properly based on rotation
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > const.ARENA_WIDTH:
            self.rect.right = const.ARENA_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= const.ARENA_HEIGHT:
            self.rect.bottom = const.ARENA_HEIGHT

    def _debug_rotatecenterarena(self, blnClockwise):
        tplCenterArena = (const.ARENA_WIDTH / 2, const.ARENA_HEIGHT / 2)
        lngRotRadius = 16
        if blnClockwise:
            self.dblRotation -= 1
            dblRobotFacing = -90
        else:
            self.dblRotation += 1
            dblRobotFacing = 90

        # Calculate new center
        dblRadians = math.radians(self.dblRotation)
        tplNewCenter = (
            tplCenterArena[0] + lngRotRadius * math.cos(dblRadians),
            tplCenterArena[1] - lngRotRadius * math.sin(dblRadians)
        )

        # first, rotate in place
        self.surf = pygame.transform.rotate(self.surfBase, self.dblRotation - self.dblInitialImageOffset + dblRobotFacing)
        self.rect = self.surf.get_rect(center=tplNewCenter)

    sbln_MoveAngularRoundUp = False
    # TODO probly want to redo this with quaternions, if possible
    def _move_angular(self, lngAngularVel, tplCenterRot, dblTrackToCenterAngleAdj):
        self.dblRotation += lngAngularVel
        self.dblRotation %= 360
        if self.dblRotation < 0:
            self.dblRotation += 360

        if tplCenterRot == self.rect.center:
            tplNewCenter = tplCenterRot
        else:
            dblRadians = math.radians(self.dblRotation + dblTrackToCenterAngleAdj)

            dblNewCenterX = tplCenterRot[0] + const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.cos(dblRadians)
            dblNewCenterY = tplCenterRot[1] - const.CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER * math.sin(dblRadians)
            lngNewCenterX = int(dblNewCenterX)
            lngNewCenterY = int(dblNewCenterY)

            # Alternate rounding up and down so we don't accumulate up/left shifts
            if self._move_angular_roundup_toggle:
                self._move_angular_roundup_toggle = False
                lngNewCenterX += 1
                lngNewCenterY += 1
            else:
                self._move_angular_roundup_toggle = True

            # # Remainder approach - something is wrong with this, though
            # self._move_angular_remainder_x = dblNewCenterX - lngNewCenterX
            # self._move_angular_remainder_y = dblNewCenterY - lngNewCenterY

            # if we constantly round down, center of mass will shift up/left each rotation
            # if self._move_angular_remainder_x > 1:
            #     self._move_angular_remainder_x -= 1.0
            #     lngNewCenterX += 1
            # elif self._move_angular_remainder_x < -1:
            #     self._move_angular_remainder_x += 1.0
            #     lngNewCenterX -= 1
            #
            # if self._move_angular_remainder_y > 1:
            #     self._move_angular_remainder_y -= 1.0
            #     lngNewCenterY += 1
            # elif self._move_angular_remainder_y < -1:
            #     self._move_angular_remainder_y += 1.0
            #     lngNewCenterY -= 1

            tplNewCenter = (lngNewCenterX, lngNewCenterY)

        self.surf = pygame.transform.rotate(self.surfBase, self.dblRotation - self.dblInitialImageOffset)
        self.rect = self.surf.get_rect(center=tplNewCenter)
        # self.rect = self.surf.get_rect(center=self.rect.center)

        # If we rotate into the wall... fuck it, just pop ourselves out for now
        # TODO probly don't actually do this though
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > const.ARENA_WIDTH:
            self.rect.right = const.ARENA_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= const.ARENA_HEIGHT:
            self.rect.bottom = const.ARENA_HEIGHT
