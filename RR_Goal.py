import pygame
import RR_Constants as const
from typing import List
from RR_Ball import Ball
from MyUtils import Div0

class Goal(pygame.sprite.Sprite):
    def __init__(self, intTeam):
        super(Goal, self).__init__()
        self.intTeam = intTeam
        self.surf = pygame.Surface((const.GOAL_WIDTH, const.GOAL_HEIGHT))
        if intTeam == const.TEAM_HAPPY:
            self.tplColor = const.COLOR_GOAL_HAPPY
            # Bottom right
            self.tplCenter = (const.ARENA_WIDTH - const.GOAL_WIDTH/2,
                              const.ARENA_HEIGHT - const.GOAL_HEIGHT/2)
            self.lstPoly = [(const.GOAL_WIDTH, 0),
                            (0, const.GOAL_HEIGHT),
                            (const.GOAL_WIDTH, const.GOAL_HEIGHT)]
        else:
            self.tplColor = const.COLOR_GOAL_GRUMPY
            self.tplCenter = (const.GOAL_WIDTH/2, const.GOAL_HEIGHT/2)
            self.lstPoly = [(const.GOAL_WIDTH, 0),
                            (0, const.GOAL_HEIGHT),
                            (0,0)]

        self.rect = self.surf.get_rect(center=self.tplCenter)
        self.surf.fill(const.COLOR_BACKGROUND)
        pygame.draw.polygon(self.surf, self.tplColor, self.lstPoly)
        self.lngFrameCount = 0
        self.dctBallsCurrent = {}
        self.dctBallsPrior = {}
        self.dctPosBallsScored = {}
        self.dctNegBallsScored = {}

    def on_step_begin(self):
        self.dctBallsCurrent = {}
        self.lngFrameCount += 1

    def on_step_end(self):
        # retain the original frame count and then copy current as new prior
        for sprBall in self.dctBallsCurrent:
            if sprBall in self.dctBallsPrior:
                self.dctBallsCurrent[sprBall] = self.dctBallsPrior[sprBall]

        self.dctBallsPrior = self.dctBallsCurrent.copy()

    def track_balls(self, lstBalls: List[Ball]) -> None:
        for sprBall in lstBalls:
            if self.ball_in_goal(sprBall):
                self.dctBallsCurrent[sprBall] = self.lngFrameCount

    def ball_in_goal(self, sprBall: Ball) -> bool:
        if self.rect.left <= sprBall.rect.centerx <= self.rect.right and \
                self.rect.top <= sprBall.rect.centery <= self.rect.bottom:

            # is the center of the ball within the triangle?
            # comparing the slope of the hypotenuse to the slope of the ball will
            # tell us if it's in the top left or bottom right portion
            dblSlopeHypotenuse = Div0(self.rect.bottom - self.rect.centery, self.rect.centerx - self.rect.left)
            dblSlopeBall = Div0(self.rect.bottom - sprBall.rect.centery, sprBall.rect.centerx - self.rect.left)

            if self.intTeam == const.TEAM_HAPPY:
                return dblSlopeBall <= dblSlopeHypotenuse
            else:
                return dblSlopeBall >= dblSlopeHypotenuse

    def update_score(self) -> List[Ball]:
        lstConsumedBalls = []
        for sprBall in self.dctBallsCurrent:
            if sprBall in self.dctBallsPrior and \
                    self.lngFrameCount - self.dctBallsPrior[sprBall] >= const.TIME_BALL_IN_GOAL_STEPS:
                lstConsumedBalls.append(sprBall)
                if sprBall.is_positive():
                    self.dctPosBallsScored[sprBall] = self.lngFrameCount
                else:
                    self.dctNegBallsScored[sprBall] = self.lngFrameCount

        return lstConsumedBalls

    def get_score(self):
        return len(self.dctPosBallsScored) * const.POINTS_BALL - len(self.dctNegBallsScored) * const.POINTS_BALL

    def is_destroyed(self) -> bool:
        return len(self.dctNegBallsScored) >= const.MAX_NEG_BALLS


