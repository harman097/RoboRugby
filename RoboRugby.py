from __future__ import absolute_import, division, print_function
from typing import List, Dict, Tuple
import MyUtils
from MyUtils import Stage
import gym
import pygame
import math
import RR_Constants as const
from enum import Enum

Stage("Initialize pygame")
pygame.init()
mScreen = pygame.display.set_mode((const.ARENA_WIDTH, const.ARENA_HEIGHT))

from RR_Ball import Ball
from RR_Robot import Robot
from RR_Goal import Goal
import RR_TrashyPhysics as TrashyPhysics

MyUtils.PRINT_STAGE = False  # Disable stage spam

# TODO perf stuff (if we end up doing serious training with this)
# (1) Pre-cache sin/cos results in a dictionary when initializing -or- quaternions? (they're a thing, idk the math tho)
# Avoid math.sin/cos and radian->degree conversion (cuz it's probly slow?)
# (2) Can avoid pygame.transform.rotate calls entirely if we're not rendering (I'm assuming it's slow)

# TODO properly inherit
class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        Stage("Initializing RoboRugby...")
        self.grpAllSprites = pygame.sprite.Group()
        self.lngStepCount = 0

        # left wall, right wall, top wall, bottom wall - 100 px buffer
        self.lstWalls = [
            pygame.Rect(-100, -100, 100, const.ARENA_HEIGHT + 200),
            pygame.Rect(const.ARENA_WIDTH, -100, 100, const.ARENA_HEIGHT + 200),
            pygame.Rect(-100, -100, const.ARENA_WIDTH + 200, 100),
            pygame.Rect(const.ARENA_HEIGHT, -100, const.ARENA_WIDTH + 200, 100)
        ]

        # TODO set these properly (if you want to properly implement gym.Env)
        # self.action_space = gym.spaces.Tuple()? .MultiDiscrete()
        # self.observation_space = gym.spaces.MultiDiscrete()? Other?

        Stage("Here are the goals")
        self.sprHappyGoal = Goal(const.TEAM_HAPPY)
        self.sprGrumpyGoal = Goal(const.TEAM_GRUMPY)
        self.grpAllSprites.add(self.sprHappyGoal)
        self.grpAllSprites.add(self.sprGrumpyGoal)

        Stage("Bring out ze Robots!")
        self.grpRobots = pygame.sprite.Group()
        lngTotalRobots = const.NUM_ROBOTS_GRUMPY + const.NUM_ROBOTS_HAPPY
        self.lstRobots = [0] * (lngTotalRobots)
        for i in range(const.NUM_ROBOTS_HAPPY):
            self.lstRobots[i] = Robot(const.TEAM_HAPPY)
            while pygame.sprite.spritecollideany(self.lstRobots[i], self.grpAllSprites):
                # Keep trying random start positions until there's no overlap
                self.lstRobots[i] = Robot(const.TEAM_HAPPY)
            self.grpRobots.add(self.lstRobots[i])
            self.grpAllSprites.add(self.lstRobots[i])

        for i in range(const.NUM_ROBOTS_HAPPY, lngTotalRobots):
            self.lstRobots[i] = Robot(const.TEAM_GRUMPY)
            while pygame.sprite.spritecollideany(self.lstRobots[i], self.grpAllSprites):
                # Keep trying random start positions until there's no overlap
                self.lstRobots[i] = Robot(const.TEAM_GRUMPY)
            self.grpRobots.add(self.lstRobots[i])
            self.grpAllSprites.add(self.lstRobots[i])

        Stage("Give it some balls!")
        self.grpBalls = pygame.sprite.Group()
        self._lstPosBalls = [None] * const.NUM_BALL_POS  # type: List[Ball]
        self._lstNegBalls = [None] * const.NUM_BALL_POS  # type: List[Ball]
        for i in range(const.NUM_BALL_POS):
            while True:  # Just make one until we have no initial collisions
                objNewBall = Ball(const.COLOR_BALL_POS)
                if pygame.sprite.spritecollideany(objNewBall, self.grpAllSprites) is None:
                    break
            self.grpBalls.add(objNewBall)
            self.grpAllSprites.add(objNewBall)
            self._lstPosBalls[i] = objNewBall

        for i in range(const.NUM_BALL_NEG):
            while True:  # Just make one until we have no initial collisions
                objNewBall = Ball(const.COLOR_BALL_NEG)
                if pygame.sprite.spritecollideany(objNewBall, self.grpAllSprites) is None:
                    break
            self.grpBalls.add(objNewBall)
            self.grpAllSprites.add(objNewBall)
            self._lstNegBalls[i] = objNewBall

        self.dblBallDistSum = self._get_ball_distance_sum()

    def reset(self):
        raise NotImplementedError("Probly just move most of the logic from __init__ to here")
        return self._get_game_state()

    def render(self, mode='human'):

        if mode != 'human':
            raise NotImplementedError("Other mode types not supported.")

        Stage("Render RoboRugby!")
        # Clear the screen
        mScreen.fill(const.COLOR_BACKGROUND)

        pygame.draw.circle(mScreen, (255,255,0), (400,400), 200)

        # Redraw each sprite in their relative "z" order
        mScreen.blit(self.sprHappyGoal.surf, self.sprHappyGoal.rect)
        mScreen.blit(self.sprGrumpyGoal.surf, self.sprGrumpyGoal.rect)

        for objSprite in self.grpRobots:
            mScreen.blit(objSprite.surf, objSprite.rect)

        for objSprite in self.grpBalls:
            mScreen.blit(objSprite.surf, objSprite.rect)

        # update the display
        pygame.display.flip()

    def step(self, lstArgs: List[Tuple]):
        if self.game_is_done():
            raise Exception("Game is over. Go home.")

        Stage("Initialize")
        self.lngStepCount += 1
        dblGrumpyScore = -1 * const.POINTS_TIME_PENALTY
        dblHappyScore = -1 * const.POINTS_TIME_PENALTY
        setNaughtyRobots = set()

        Stage("Prep ze sprites!")
        for sprSprite in self.grpAllSprites:
            if hasattr(sprSprite, 'on_step_begin'):
                sprSprite.on_step_begin()

        Stage("Activate robot engines!")
        if len(lstArgs) > const.NUM_ROBOTS_TOTAL:
            raise Exception(f"{len(lstArgs)} commands but only {const.NUM_ROBOTS_TOTAL} robots.")
        for i in range(len(lstArgs)):
            self.lstRobots[i].set_thrust(lstArgs[i][0], lstArgs[i][1])

        """
        New pseudo, now that move_one() is viable
        For _ in range(const.ROBOT_SPEED) (aka pixels to move per frame)
            robots.move_one()
            for each robot/robot collision
                undo both of their latest moves
                set movement speed to 0
            for each robot/ball collision
                impart velocity to the ball
            balls.move_one()
            for each ball/ball collision
                bounce balls off each other
            for each robot/ball collision
                undo the robot movement (it hit the wall or another robot)        
        """

        for _ in range(const.MOVES_PER_FRAME):

            # Typehint all the loop variables we're going to use
            # so Pycharm will make our lives easier
            sprRobot: Robot
            sprRobot1: Robot
            sprRobot2: Robot
            sprBall: Ball
            sprBall1: Ball
            sprBall2: Ball

            Stage("Move those robots!")
            for sprRobot in self.grpRobots.sprites():
                sprRobot.move()

            Stage("Did they smash each other?")
            for sprRobot1, sprRobot2 in TrashyPhysics.collision_pairs_self(
                    self.grpRobots, fncCollided=TrashyPhysics.robots_collided):
                # TODO logic on "who is at fault?" probly needs to be a bit better...
                if sprRobot1.lngRThrust != 0 or sprRobot1.lngLThrust != 0:
                    setNaughtyRobots.add(sprRobot1)
                if sprRobot2.lngRThrust != 0 or sprRobot2.lngLThrust != 0:
                    setNaughtyRobots.add(sprRobot2)
                lngNaughtyLoop = 0
                while True:
                    lngNaughtyLoop += 1
                    sprRobot1.undo_move()
                    sprRobot2.undo_move()
                    if not TrashyPhysics.robots_collided(sprRobot1, sprRobot2):
                        break
                    elif lngNaughtyLoop > 1000:
                        raise Exception("Dude this is seriously broken")

            Stage("Push those balls!")
            for sprBall, sprRobot in TrashyPhysics.collision_pairs(
                    self.grpBalls, self.grpRobots, fncCollided=TrashyPhysics.ball_robot_collided):
                TrashyPhysics.apply_force_to_ball(sprRobot, sprBall)

            Stage("Roll the balls!")
            for sprBall in self.grpBalls.sprites():
                sprBall.move()

            Stage("Bounce the balls!")
            for sprBall1, sprBall2 in TrashyPhysics.collision_pairs_self(
                    self.grpBalls, fncCollided=TrashyPhysics.balls_collided):
                TrashyPhysics.bounce_balls(sprBall1, sprBall2)

            Stage("Are robots still being naughty? Deny movement.")
            for sprBall, sprRobot in TrashyPhysics.collision_pairs(
                    self.grpBalls, self.grpRobots, fncCollided=TrashyPhysics.ball_robot_collided):
                sprRobot.undo_move()

        Stage("Flag balls in the goal")
        self.sprHappyGoal.track_balls(self.grpBalls.sprites())
        self.sprGrumpyGoal.track_balls(self.grpBalls.sprites())
        
        Stage("Flag robots in goals")
        for sprRobot in self.lstRobots:
            if TrashyPhysics.robot_in_goal(sprRobot, self.sprHappyGoal) or \
                    TrashyPhysics.robot_in_goal(sprRobot, self.sprGrumpyGoal):
                if sprRobot.intTeam == const.TEAM_HAPPY:
                    dblHappyScore -= const.POINTS_ROBOT_IN_GOAL_PENALTY
                else:
                    dblGrumpyScore -= const.POINTS_ROBOT_IN_GOAL_PENALTY

        Stage("Commit balls that have scored")
        dblScoreDelta = 0  # positive = good for Happy, negative = good for Grumpy
        
        for sprBall in self.sprHappyGoal.update_score():
            if sprBall.tplColor == const.COLOR_BALL_POS:
                dblScoreDelta += const.POINTS_BALL_SCORED
            else:
                dblScoreDelta -= const.POINTS_BALL_SCORED
            sprBall.kill()
            
        for sprBall in self.sprGrumpyGoal.update_score():
            if sprBall.tplColor == const.COLOR_BALL_POS:
                dblScoreDelta -= const.POINTS_BALL_SCORED
            else:
                dblScoreDelta += const.POINTS_BALL_SCORED
            sprBall.kill()
            
        dblHappyScore += dblScoreDelta
        dblGrumpyScore -= dblScoreDelta

        Stage("Handle any end of step activities")
        for sprSprite in self.grpAllSprites:
            if hasattr(sprSprite, 'on_step_end'):
                sprSprite.on_step_end()

        Stage("Who was naughty?")
        for sprNaughtyBot in setNaughtyRobots:
            if sprNaughtyBot.intTeam == const.TEAM_HAPPY:
                dblHappyScore -= const.POINTS_ROBOT_CRASH_PENALTY
            else:
                dblGrumpyScore -= const.POINTS_ROBOT_CRASH_PENALTY

        Stage("Award points for ball travel")
        dblBallDistSumStart = self.dblBallDistSum
        self.dblBallDistSum = self._get_ball_distance_sum()
        dblDelta = const.POINTS_BALL_TRAVEL_MULT * (self.dblBallDistSum - dblBallDistSumStart)
        dblHappyScore += dblDelta  # ball travel points are zero-sum
        dblGrumpyScore -= dblDelta

        Stage("Anybody win?")
        if self.sprHappyGoal.is_destroyed():
            dblGrumpyScore += const.POINTS_GOAL_DESTROYED
            dblHappyScore -= const.POINTS_GOAL_DESTROYED

        if self.sprGrumpyGoal.is_destroyed():
            dblHappyScore += const.POINTS_GOAL_DESTROYED
            dblGrumpyScore -= const.POINTS_GOAL_DESTROYED

        return self._get_game_state(), dblHappyScore, self.game_is_done(), GameEnv.DebugInfo(dblGrumpyScore)

    def _get_game_state(self, intTeam=const.TEAM_HAPPY):
        # TODO this
        return 1

    def game_is_done(self):
        return self.lngStepCount > const.GAME_LENGTH_STEPS or \
               self.sprGrumpyGoal.is_destroyed() or \
               self.sprHappyGoal.is_destroyed() or \
               not bool(self.grpBalls)
    
    def _get_ball_distance_sum(self) -> float:
        """ Distance meaning 'Distance from Grumpy's goal'"""
        def _dist(sprBall:Ball)->float:
            return math.pow(sprBall.rectDbl.centerx**2 + sprBall.rectDbl.centery**2, .5)
        
        # Higher score the farther away positive balls get from Grumpy's goal (and therefore closer to Happy's goal)
        # Lower score the farther away negative balls get from Grumpy's goal (and therefore closer to Happy's goal)
        return sum(map(_dist, self._lstPosBalls)) - sum(map(_dist, self._lstNegBalls))

    # Add more things here as they come up
    class DebugInfo:
        def __init__(self, dblGrumpyScore:float):
            self.dblGrumpyScore = dblGrumpyScore  #type: float







