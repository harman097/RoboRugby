import RR_Constants as const
import pygame
import random
from MyUtils import FloatRect

class Ball(pygame.sprite.Sprite):
    def __init__(self, tplColor):
        super(Ball, self).__init__()
        self.blnHitWall = False
        self.dblXVelocity = 0
        self.dblYVelocity = 0
        self.tplColor = tplColor
        # self.surf = mScreen
        self.surf = pygame.Surface((const.BALL_RADIUS*2, const.BALL_RADIUS*2))
        self.surf.fill((255,255,255))
        self.surf.set_colorkey((255,255,255))
        tplCenter = (  # Make sure the robot can get it off the wall
                random.randint(const.ROBOT_WIDTH, const.ARENA_WIDTH - const.ROBOT_WIDTH),
                random.randint(const.ROBOT_WIDTH, const.ARENA_HEIGHT - const.ROBOT_WIDTH)
            )
        # self.rect = self.surf.get_rect(center=(tplCenter))

        # integer rect, used by pygame for rendering
        self.rect = self.surf.get_rect(center=tplCenter)
        self.radius = const.BALL_RADIUS  # used by pygame to check circle collisions
        pygame.draw.circle(self.surf, self.tplColor, (const.BALL_RADIUS, const.BALL_RADIUS), const.BALL_RADIUS)

        # actual rect, used internally to calculate location
        self.dblRect = FloatRect(self.rect.left, self.rect.right, self.rect.top, self.rect.bottom)
        self.dblRectPrior = self.dblRect.copy()

    def on_step_begin(self):
        self.dblRectPrior = self.dblRect.copy()
        self.blnHitWall = False

    def on_step_end(self):
        print("Ball.on_step_end() not implemented.")

    def move(self):
        self.dblRect.left += self.dblXVelocity
        self.dblRect.top += self.dblYVelocity

        # Bounce the ball off the wall
        # "K_Wall" = how much to dampen the bounce
        if self.dblRect.left < 0:
            self.dblRect.left *= -1 * const.BOUNCE_K_WALL
            self.dblXVelocity *= -1 * const.BOUNCE_K_WALL
        if self.dblRect.right > const.ARENA_WIDTH:
            self.dblRect.right = const.ARENA_WIDTH - (self.dblRect.right - const.ARENA_WIDTH) * const.BOUNCE_K_WALL
            self.dblXVelocity *= -1 * const.BOUNCE_K_WALL
        if self.dblRect.top <= 0:
            self.dblRect.top *= -1 * const.BOUNCE_K_WALL
            self.dblYVelocity *= -1 * const.BOUNCE_K_WALL
        if self.dblRect.bottom >= const.ARENA_HEIGHT:
            self.dblRect.bottom = const.ARENA_HEIGHT - (self.dblRect.bottom - const.ARENA_WIDTH) * const.BOUNCE_K_WALL
            self.dblYVelocity *= -1 * const.BOUNCE_K_WALL

        self.rect.left = int(self.dblRect.left)
        self.rect.top = int(self.dblRect.top)

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