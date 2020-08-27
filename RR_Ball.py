import RR_Constants as const
import pygame
import random

class Ball(pygame.sprite.Sprite):
    def __init__(self, tplColor):
        super(Ball, self).__init__()
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

        self.rect = self.surf.get_rect(center=tplCenter)
        pygame.draw.circle(self.surf, self.tplColor, (const.BALL_RADIUS, const.BALL_RADIUS), const.BALL_RADIUS)
        self.rectPrior = self.rect.copy()

    def move(self):
        self.rectPrior = self.rect.copy()
        self.rect.move_ip(self.dblXVelocity, self.dblYVelocity)

        # Bounce the ball off the wall
        # "K_Wall" = how much to dampen the bounce
        if self.rect.left < 0:
            self.rect.left *= -1 * const.BOUNCE_K_WALL
            self.dblXVelocity *= -1 * const.BOUNCE_K_WALL
        if self.rect.right > const.ARENA_WIDTH:
            self.rect.right = const.ARENA_WIDTH - (self.rect.right - const.ARENA_WIDTH) * const.BOUNCE_K_WALL
            self.dblXVelocity *= -1 * const.BOUNCE_K_WALL
        if self.rect.top <= 0:
            self.rect.top *= -1 * const.BOUNCE_K_WALL
            self.dblYVelocity *= -1 * const.BOUNCE_K_WALL
        if self.rect.bottom >= const.ARENA_HEIGHT:
            self.rect.bottom = const.ARENA_HEIGHT - (self.rect.bottom - const.ARENA_WIDTH) * const.BOUNCE_K_WALL
            self.dblYVelocity *= -1 * const.BOUNCE_K_WALL

        self.dblXVelocity *= const.BALL_SLOWDOWN
        self.dblYVelocity *= const.BALL_SLOWDOWN

    def set_velocity(self, dblXVelocity, dblYVelocity):
        self.dblXVelocity = dblXVelocity
        self.dblYVelocity = dblYVelocity

    def is_positive(self):
        return self.tplColor == const.COLOR_BALL_POS