import robo_rugby.gym_env.RR_Constants as const
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
    def radius(self) -> float:  # used by pygame to check circle collisions
        return const.BALL_RADIUS

    def __init__(self, tplColor:Tuple[int,int,int], tplCenter:Tuple[float,float]):
        super(Ball, self).__init__()
        self.dblXVelocity = 0
        self.dblYVelocity = 0
        self.tplColor = tplColor
        # self.surf = mScreen
        self.surf = pygame.Surface((const.BALL_RADIUS*2, const.BALL_RADIUS*2))
        self.surf.fill((255,255,255))
        self.surf.set_colorkey((255,255,255))
        pygame.draw.circle(self.surf, self.tplColor, (const.BALL_RADIUS, const.BALL_RADIUS), const.BALL_RADIUS)

        self.rectDbl = FloatRect(0, self.surf.get_width(), 0, self.surf.get_height())
        self.rectDbl.center = tplCenter
        self.rectDblPriorStep = self.rectDbl.copy()
        self.lngFrameMass = const.MASS_BALL

    def on_step_begin(self):
        self.rectDblPriorStep = self.rectDbl.copy()
        self.lngFrameMass = const.MASS_BALL

    def on_reset(self):
        # avoid calling __init__ for now, just so we don't redraw the surface
        # if ball ever gets more complicated, though, can just call __init__
        self.dblXVelocity = 0
        self.dblYVelocity = 0
        self.lngFrameMass = const.MASS_BALL
        self.rectDblPriorStep = self.rectDbl.copy()

    def move(self):
        self.rectDbl.left += self.dblXVelocity
        self.rectDbl.top += self.dblYVelocity

        # Bounce the ball off the wall
        # "K_Wall" = how much to dampen the bounce
        if self.rectDbl.left < 0:
            self.rectDbl.left *= -1 * const.BOUNCE_K_WALL
            self.dblXVelocity *= -1 * const.BOUNCE_K_WALL
            self.lngFrameMass = const.MASS_WALL
        if self.rectDbl.right > const.ARENA_WIDTH:
            self.rectDbl.right = const.ARENA_WIDTH - (self.rectDbl.right - const.ARENA_WIDTH) * const.BOUNCE_K_WALL
            self.dblXVelocity *= -1 * const.BOUNCE_K_WALL
            self.lngFrameMass = const.MASS_WALL
        if self.rectDbl.top <= 0:
            self.rectDbl.top *= -1 * const.BOUNCE_K_WALL
            self.dblYVelocity *= -1 * const.BOUNCE_K_WALL
            self.lngFrameMass = const.MASS_WALL
        if self.rectDbl.bottom >= const.ARENA_HEIGHT:
            self.rectDbl.bottom = const.ARENA_HEIGHT - (self.rectDbl.bottom - const.ARENA_WIDTH) * const.BOUNCE_K_WALL
            self.dblYVelocity *= -1 * const.BOUNCE_K_WALL
            self.lngFrameMass = const.MASS_WALL

        self.rect.left = int(self.rectDbl.left)
        self.rect.top = int(self.rectDbl.top)

        self.dblXVelocity *= const.BALL_SLOWDOWN
        self.dblYVelocity *= const.BALL_SLOWDOWN
        if abs(self.dblXVelocity) < const.BALL_MIN_SPEED:
            self.dblXVelocity = 0
        if abs(self.dblYVelocity) < const.BALL_MIN_SPEED:
            self.dblYVelocity = 0

    def set_velocity(self, dblXVelocity, dblYVelocity):
        self.dblXVelocity = dblXVelocity
        self.dblYVelocity = dblYVelocity

    def is_positive(self):
        return self.tplColor == const.COLOR_BALL_POS