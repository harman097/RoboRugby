import pygame
import RR_Constants as const
from typing import List
from RR_Ball import Ball
from MyUtils import Div0, RightTriangle, Point

class Goal(pygame.sprite.Sprite):
    def __init__(self, intTeam):
        super(Goal, self).__init__()
        self.intTeam = intTeam
        self.surf = pygame.Surface((const.GOAL_WIDTH, const.GOAL_HEIGHT))
        self.surf.fill(const.COLOR_BACKGROUND)

        if intTeam == const.TEAM_HAPPY:
            self.tplColor = const.COLOR_GOAL_HAPPY
            # Bottom right
            self.triShape = RightTriangle(
                Point(const.ARENA_WIDTH, const.ARENA_HEIGHT),
                Point(const.ARENA_WIDTH, const.ARENA_HEIGHT - const.GOAL_HEIGHT),
                Point(const.ARENA_WIDTH - const.GOAL_WIDTH, const.ARENA_HEIGHT))
        else:
            self.tplColor = const.COLOR_GOAL_GRUMPY
            # Top left
            self.triShape = RightTriangle(
                Point(const.GOAL_WIDTH, 0),
                Point(0, const.GOAL_HEIGHT),
                Point(0, 0)
            )

        self.rect = self.surf.get_rect(
            center=(
                (self.triShape.left + self.triShape.right)/2,
                (self.triShape.top + self.triShape.bottom)/2
            )
        )

        pygame.draw.polygon(
            self.surf, self.tplColor,
            list(map(lambda tplI: (tplI[0] - self.triShape.left, tplI[1] - self.triShape.top), self.triShape.corners)))

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
        return self.triShape.contains_point(sprBall.rectDbl.center)

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
        return len(self.dctPosBallsScored) * const.POINTS_BALL_SCORED - len(self.dctNegBallsScored) * const.POINTS_BALL_SCORED

    def is_destroyed(self) -> bool:
        return len(self.dctNegBallsScored) >= const.MAX_NEG_BALLS


