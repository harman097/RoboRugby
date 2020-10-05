from . import RR_Constants as const
import pygame
from typing import Tuple
from MyUtils import FloatRect

class Ball(pygame.sprite.Sprite):

    @property
    def rect(self) -> pygame.Rect:  # used by pygame for rendering (otherwise use
        return pygame.Rect(
            self.rectDbl.left,
            self.rectDbl.top,
            self.rectDbl.right - self.rectDbl.left,
            self.rectDbl.bottom - self.rectDbl.top
        )

    @property
    def tplColor(self) -> Tuple[float,float,float]:
        return self._tplColor

    @tplColor.setter
    def tplColor(self, new_color):
        self._tplColor = new_color
        self.surf.fill((255, 255, 255))
        self.surf.set_colorkey((255, 255, 255))
        pygame.draw.circle(self.surf, self._tplColor, (const.BALL_RADIUS, const.BALL_RADIUS), const.BALL_RADIUS)

    @property
    def radius(self) -> float:  # used by pygame to check circle collisions
        return const.BALL_RADIUS

    @property
    def is_positive(self):
        return self.tplColor == const.COLOR_BALL_POS

    @property
    def is_negative(self):
        return self.tplColor == const.COLOR_BALL_NEG

    def __init__(self, tplColor:Tuple[int,int,int], tplCenter:Tuple[float,float]):
        super(Ball, self).__init__()
        self.dbl_velocity_x = 0
        self.dbl_velocity_y = 0
        self.dbl_force_x = 0
        self.dbl_force_y = 0
        self.bln_moved_cur_frame = False
        self._tplColor = tplColor
        # self.surf = mScreen
        self.surf = pygame.Surface((const.BALL_RADIUS*2, const.BALL_RADIUS*2))
        self.surf.fill((255,255,255))
        self.surf.set_colorkey((255,255,255))
        pygame.draw.circle(self.surf, self._tplColor, (const.BALL_RADIUS, const.BALL_RADIUS), const.BALL_RADIUS)

        self.rectDbl = FloatRect(0, self.surf.get_width(), 0, self.surf.get_height())
        self.rectDbl.center = tplCenter
        self.rectDblPriorStep = self.rectDbl.copy()
        self.rectDblPriorFrame = self.rectDbl.copy()
        self.lngFrameMass = const.MASS_BALL

    def on_step_begin(self):
        self.rectDblPriorStep = self.rectDbl.copy()

    def on_frame_begin(self):
        self.lngFrameMass = const.MASS_BALL
        self.dbl_force_x = 0  # Force is applied each frame
        self.dbl_force_y = 0
        self.bln_moved_cur_frame = False
        self.rectDblPriorFrame = self.rectDbl.copy()

    def on_reset(self):
        # avoid calling __init__ for now, just so we don't redraw the surface
        # if ball ever gets more complicated, though, can just call __init__
        self.dbl_velocity_x = 0
        self.dbl_velocity_y = 0
        self.lngFrameMass = const.MASS_BALL
        self.rectDblPriorStep = self.rectDbl.copy()

    def move(self):
        self.bln_moved_cur_frame = True

        # We're just translating force directly to velocity like the trash that we are
        if self.dbl_velocity_x >= 0 and self.dbl_force_x >= 0:
            self.dbl_velocity_x = max(self.dbl_velocity_x, self.dbl_force_x)
        elif self.dbl_velocity_x <= 0 and self.dbl_force_x <= 0:
            self.dbl_velocity_x = min(self.dbl_velocity_x, self.dbl_force_x)
        else:  # should only happen with multi-collisions
            self.dbl_velocity_x += self.dbl_force_x

        if self.dbl_velocity_y >= 0 and self.dbl_force_y >= 0:
            self.dbl_velocity_y = max(self.dbl_velocity_y, self.dbl_force_y)
        elif self.dbl_velocity_y <= 0 and self.dbl_force_y <= 0:
            self.dbl_velocity_y = min(self.dbl_velocity_y, self.dbl_force_y)
        else:  # should only happen with multi-collisions
            self.dbl_velocity_y += self.dbl_force_y

        self.rectDbl.left += self.dbl_velocity_x
        self.rectDbl.top += self.dbl_velocity_y
        self.rect.left = int(self.rectDbl.left)
        self.rect.top = int(self.rectDbl.top)
        self.dbl_velocity_x *= const.BALL_SLOWDOWN
        self.dbl_velocity_y *= const.BALL_SLOWDOWN
        if abs(self.dbl_velocity_x) < const.BALL_MIN_SPEED:
            self.dbl_velocity_x = 0
        if abs(self.dbl_velocity_y) < const.BALL_MIN_SPEED:
            self.dbl_velocity_y = 0

    def undo_move(self) -> bool:
        if self.bln_moved_cur_frame:
            self.bln_moved_cur_frame = False
            self.rectDbl = self.rectDblPriorFrame.copy()
            return True
        else:
            return False

    def set_velocity(self, dblXVelocity, dblYVelocity):
        self.dbl_velocity_x = dblXVelocity
        self.dbl_velocity_y = dblYVelocity
