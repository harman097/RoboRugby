from __future__ import absolute_import, division, print_function
from typing import List, Tuple, Set
import MyUtils
from MyUtils import Stage, Point, FloatRect, distance, angle_degrees, angle_radians, Div0, get_slope_yint, get_line_intersection
import gym
from gym.utils import seeding
import pygame
import math
import random
import robo_rugby.gym_env.RR_Constants as const
import numpy as np
from enum import Enum
from PIL import Image

Stage("Initialize pygame")
pygame.init()
mScreen = pygame.display.set_mode((const.ARENA_WIDTH, const.ARENA_HEIGHT))

from robo_rugby.gym_env.RR_Ball import Ball
from robo_rugby.gym_env.RR_Robot import Robot
from robo_rugby.gym_env.RR_Goal import Goal
import robo_rugby.gym_env.RR_TrashyPhysics as TrashyPhysics

MyUtils.PRINT_STAGE = False  # Disable stage spam


# TODO perf stuff (if we end up doing serious training with this)
# (1) Pre-cache sin/cos results in a dictionary when initializing -or- quaternions? (they're a thing, idk the math tho)
# Avoid math.sin/cos and radian->degree conversion (cuz it's probly slow?)
# (2) Can avoid pygame.transform.rotate calls entirely if we're not rendering (I'm assuming it's slow)

class GameEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': const.FRAMERATE
    }
    _h5 = const.ARENA_HEIGHT / 5
    _w5 = const.ARENA_WIDTH / 5
    CONFIG_RANDOM = None
    CONFIG_STANDARD = [
        [
            (const.ARENA_WIDTH / 2 + 1 * _w5, const.ARENA_HEIGHT - 1 * _h5, 135),
            (const.ARENA_WIDTH / 2 + 2 * _w5, const.ARENA_HEIGHT - 2 * _h5, 135),
            (const.ARENA_WIDTH / 2 - 1 * _w5, 1 * _h5, 315),
            (const.ARENA_WIDTH / 2 - 2 * _w5, 2 * _h5, 315)
        ],
        [
            (_w5 * 1, const.ARENA_HEIGHT - _h5 * 1),
            (_w5 * 2, const.ARENA_HEIGHT - _h5 * 2),
            (_w5 * 3, const.ARENA_HEIGHT - _h5 * 3),
            (_w5 * 4, const.ARENA_HEIGHT - _h5 * 4),
            (const.ARENA_WIDTH / 2, _h5),
            (const.ARENA_WIDTH / 2, const.ARENA_HEIGHT - _h5),
            (_w5, const.ARENA_HEIGHT / 2),
            (const.ARENA_WIDTH - _w5, const.ARENA_HEIGHT / 2)
        ]
    ]

    @property
    def lstHappyBots(self) -> List[Robot]:  # Happy bots in 1st half of list
        return self.lstRobots[:const.NUM_ROBOTS_HAPPY]

    @property
    def lstGrumpyBots(self) -> List[Robot]:  # Grumpy bots in 2nd half of list
        return self.lstRobots[const.NUM_ROBOTS_HAPPY:]

    @property
    def lstPosBalls(self) -> List[Ball]:  # Pos balls in 1st half of list
        return self.lstBalls[:const.NUM_BALL_POS]

    @property
    def lstNegBalls(self) -> List[Ball]:  # Pos balls in 1st half of list
        return self.lstBalls[const.NUM_BALL_POS:]

    def __init__(self, lst_starting_config: List[Tuple[float, float]] = CONFIG_RANDOM):
        Stage("Initializing RoboRugby...")
        self.grpAllSprites = pygame.sprite.Group()
        self.lngStepCount = 0

        Stage("Here are the goals")
        self.sprHappyGoal = Goal(const.TEAM_HAPPY)
        self.sprGrumpyGoal = Goal(const.TEAM_GRUMPY)
        self.grpAllSprites.add(self.sprHappyGoal)
        self.grpAllSprites.add(self.sprGrumpyGoal)

        Stage("Bring out ze Robots!")
        self.grpRobots = pygame.sprite.Group()
        lngTotalRobots = const.NUM_ROBOTS_GRUMPY + const.NUM_ROBOTS_HAPPY
        self.lstRobots = [None] * lngTotalRobots  # type: List[Robot]
        for i in range(const.NUM_ROBOTS_HAPPY):
            self.lstRobots[i] = Robot(const.TEAM_HAPPY, (0, 0))
            self.grpRobots.add(self.lstRobots[i])
            self.grpAllSprites.add(self.lstRobots[i])
        for i in range(const.NUM_ROBOTS_HAPPY, lngTotalRobots):
            self.lstRobots[i] = Robot(const.TEAM_GRUMPY, (0, 0))
            self.grpRobots.add(self.lstRobots[i])
            self.grpAllSprites.add(self.lstRobots[i])

        Stage("Give it some balls!")
        self.grpBalls = pygame.sprite.Group()
        lngTotalBalls = const.NUM_BALL_POS + const.NUM_BALL_NEG
        self.lstBalls = [None] * lngTotalBalls  # type: List[Ball]
        for i in range(const.NUM_BALL_POS):
            self.lstBalls[i] = Ball(const.COLOR_BALL_POS, (0, 0))
            self.grpBalls.add(self.lstBalls[i])
            self.grpAllSprites.add(self.lstBalls[i])
        for i in range(const.NUM_BALL_POS, lngTotalBalls):
            self.lstBalls[i] = Ball(const.COLOR_BALL_NEG, (0, 0))
            self.grpBalls.add(self.lstBalls[i])
            self.grpAllSprites.add(self.lstBalls[i])

        Stage("Set the starting positions")
        if lst_starting_config is None:
            self._lst_starting_positions = self._set_random_positions()
        else:
            self._lst_starting_positions = lst_starting_config
            self._set_starting_positions()

        Stage("There shall be scorekeepers! eventually")
        self.set_scorekeepers = set()  # type: Set[sk.AbstractScoreKeeper]

        # Minitaur gym is a good example of similar inputs/outputs
        # Velocity/position/rotation can never be > bounds of the arena... barring crazy constants
        if GameEnv.observation_space is None:  # shared variable from base class, gym.Env()
            arrState = self.get_game_state()
            dblObservationHigh = max(const.ARENA_WIDTH, const.ARENA_HEIGHT, 360)
            dblObservationLow = dblObservationHigh * -1
            GameEnv.observation_space = gym.spaces.Box(
                dblObservationLow, dblObservationHigh, dtype=np.float32, shape=arrState.shape)

        if GameEnv.action_space is None:  # shared variable from base class, gym.Env()
            # alngActions = np.array([1] * len(self.lstRobots) * 2)
            alngActions = np.array([1] * len(self.lstHappyBots) * 2)
            # tip: '-' operator can be applied to numpy arrays (flips each element)
            # TODO change this back to np.int
            GameEnv.action_space = gym.spaces.Box(-alngActions, alngActions, dtype=np.float32)

    def _get_positions(self):
        return [
            list(map(lambda robot: (robot.rectDbl.centerx, robot.rectDbl.centery, robot.dblRotation), self.lstRobots)),
            list(map(lambda ball: ball.rectDbl.center, self.lstBalls)),
        ]

    def _set_starting_positions(self):
        # Validate
        if self._lst_starting_positions is None:
            raise Exception("No starting configuration set.")
        if (len(self._lst_starting_positions) != 2):
            raise Exception(f"Expected list of 2 lists. Not whatever this is: {self._lst_starting_positions}")
        lst_robot_states = self._lst_starting_positions[0]
        if len(lst_robot_states) != len(self.lstRobots):
            raise Exception(f"Robot count mismatch. {len(self.lstRobots)} != {len(lst_robot_states)}.")
        lst_ball_states = self._lst_starting_positions[1]
        if len(lst_ball_states) != len(self.lstBalls):
            raise Exception(f"Ball count mismatch. {len(self.lstBalls)} != {len(lst_ball_states)}.")

        # Set robot position
        for i in range(len(self.lstRobots)):
            x, y, rot = lst_robot_states[i]  # unpack tuple
            self.lstRobots[i].rectDbl.centerx = x
            self.lstRobots[i].rectDbl.centery = y
            self.lstRobots[i].dblRotation = rot

        # Set ball position
        for i in range(len(self.lstBalls)):
            self.lstBalls[i].rectDbl.center = lst_ball_states[i]

    def _set_random_positions(self) -> List[List[Tuple]]:
        # reset all robots off map
        for spr_robot in self.lstRobots:
            spr_robot.centerx = spr_robot.centery = -1000

        for spr_robot in self.lstHappyBots:
            while True:
                # Happy team starts in the bottom-right quad with 2-robots padding
                spr_robot.rectDbl.centerx = random.randint(const.ROBOT_WIDTH * 2,
                                                           const.ARENA_WIDTH - const.ROBOT_WIDTH * 2)
                spr_robot.rectDbl.centery = random.randint(const.ROBOT_LENGTH * 2,
                                                           const.ARENA_HEIGHT - const.ROBOT_LENGTH * 2)
                spr_robot.dblRotation = random.randint(0, 360)
                if len(pygame.sprite.spritecollide(spr_robot, self.grpRobots, False)) <= 1:
                    break  # always collides with self

        for spr_robot in self.lstGrumpyBots:
            while True:
                # Grumpy team starts in the top-left quad with 2-robots padding
                spr_robot.rectDbl.centerx = random.randint(const.ROBOT_WIDTH * 2,
                                                           const.ARENA_WIDTH - const.ROBOT_WIDTH * 2)
                spr_robot.rectDbl.centery = random.randint(const.ROBOT_LENGTH * 2,
                                                           const.ARENA_HEIGHT - const.ROBOT_LENGTH * 2)
                spr_robot.dblRotation = random.randint(0, 360)
                if len(pygame.sprite.spritecollide(spr_robot, self.grpRobots, False)) <= 1:
                    break  # always collides with self

        # reset all balls off map
        for spr_ball in self.lstBalls:
            spr_ball.rectDbl.centerx = spr_ball.rectDbl.centery = -1000

        for spr_ball in self.lstPosBalls:
            while True:
                spr_ball.rectDbl.centerx = random.randint(const.ROBOT_WIDTH, const.ARENA_WIDTH - const.ROBOT_WIDTH)
                spr_ball.rectDbl.centery = random.randint(const.ROBOT_WIDTH, const.ARENA_HEIGHT - const.ROBOT_WIDTH)
                if len(pygame.sprite.spritecollide(spr_ball, self.grpAllSprites, False)) <= 1:
                    break  # always collides with self

        for spr_ball in self.lstNegBalls:
            while True:
                spr_ball.rectDbl.centerx = random.randint(const.ROBOT_WIDTH, const.ARENA_WIDTH - const.ROBOT_WIDTH)
                spr_ball.rectDbl.centery = random.randint(const.ROBOT_WIDTH, const.ARENA_HEIGHT - const.ROBOT_WIDTH)
                if len(pygame.sprite.spritecollide(spr_ball, self.grpAllSprites, False)) <= 1:
                    break  # always collides with self

        return self._get_positions()

    def reset(self, bln_randomize_pos: bool = True) -> np.ndarray:
        self.lngStepCount = 0
        for spr_ball in self.lstBalls:
            if not spr_ball.alive():
                self.grpBalls.add(spr_ball)
                self.grpAllSprites.add(spr_ball)
        for sprSprite in self.grpAllSprites:
            if hasattr(sprSprite, 'on_reset'):
                sprSprite.on_reset()
        if bln_randomize_pos:
            self._set_random_positions()
        else:
            self._set_starting_positions()

        return self.get_game_state()

    def render(self, mode='human'):

        Stage("Render RoboRugby!")
        # Clear the screen
        mScreen.fill(const.COLOR_BACKGROUND)

        pygame.draw.circle(mScreen, (255, 255, 0), (400, 400), 200)

        # Redraw each sprite in their relative "z" order
        mScreen.blit(self.sprHappyGoal.surf, self.sprHappyGoal.rect)
        mScreen.blit(self.sprGrumpyGoal.surf, self.sprGrumpyGoal.rect)

        for objSprite in self.grpRobots:
            mScreen.blit(objSprite.surf, objSprite.rect)

        for objSprite in self.grpBalls:
            mScreen.blit(objSprite.surf, objSprite.rect)

        # update the display
        pygame.display.flip()

        if mode == 'human':
            return None
        elif mode == 'rgb_array':
            return np.array(
                Image.frombytes(
                    'RGB',
                    (mScreen.get_width(), mScreen.get_height()),
                    pygame.image.tostring(mScreen, 'RGB')
                )
            )
        else:
            raise NotImplementedError("Other mode types not supported.")

    def step(self, lstArgs: List[Tuple]):
        if self.game_is_done():
            raise Exception("Game is over. Go home.")

        Stage("Initialize")
        self.lngStepCount += 1
        setBallsInGoal = set()

        self.on_step_begin()

        Stage("Activate robot engines!")
        # flatten args into 1-D array, if they're not already (tf passes like this)
        arr_args = np.concatenate(lstArgs, axis=None)
        if len(arr_args) > const.NUM_ROBOTS_TOTAL * 2:
            raise Exception(f"{len(arr_args)} commands but only {const.NUM_ROBOTS_TOTAL * 2} robot engines.")
        for i in range(0, len(arr_args), 2):
            self.lstRobots[int(i / 2)].set_thrust(arr_args[i], arr_args[i + 1])

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
                # Raise the 'on_robot_collision' event
                for scorekeeper in self.set_scorekeepers:
                    scorekeeper.on_robot_collision(sprRobot1, sprRobot2)
                lngNaughtyLoop = 0
                while True:
                    lngNaughtyLoop += 1
                    sprRobot1.undo_move()
                    sprRobot2.undo_move()
                    if not TrashyPhysics.robots_collided(sprRobot1, sprRobot2):
                        break
                    elif lngNaughtyLoop > 1000:
                        print("------ DUDE we hit the naughty loop limit... wut --------")
                        # raise Exception("Dude this is seriously broken")
                        break

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

        Stage("Flag balls in the goal")  # todo use event system
        self.sprHappyGoal.track_balls(self.grpBalls.sprites())
        self.sprGrumpyGoal.track_balls(self.grpBalls.sprites())

        # todo give them immediate points upon scoring (for now)
        for sprBall in self.lstPosBalls:

            # todo re-enable goal scoring eventually
            if self.sprHappyGoal.triShape.contains_point(sprBall.rectDbl.center):
                # dblHappyScore += const.POINTS_BALL_SCORED
                # dblGrumpyScore -= const.POINTS_BALL_SCORED
                setBallsInGoal.add(sprBall)
            elif self.sprGrumpyGoal.triShape.contains_point(sprBall.rectDbl.center):
                # dblHappyScore -= const.POINTS_BALL_SCORED
                # dblGrumpyScore += const.POINTS_BALL_SCORED
                setBallsInGoal.add(sprBall)

        for sprBall in self.lstNegBalls:
            if self.sprHappyGoal.triShape.contains_point(sprBall.rectDbl.center):
                # dblHappyScore -= const.POINTS_BALL_SCORED
                # dblGrumpyScore += const.POINTS_BALL_SCORED
                setBallsInGoal.add(sprBall)
            elif self.sprGrumpyGoal.triShape.contains_point(sprBall.rectDbl.center):
                # dblHappyScore += const.POINTS_BALL_SCORED
                # dblGrumpyScore -= const.POINTS_BALL_SCORED
                setBallsInGoal.add(sprBall)

        # todo convert this to the 'scorekeeper' system
        # todo rework the goal class a bit
        # it's kind of inconsistent in terms of division of logic between Goal vs GameEnv
        Stage("Commit balls that have scored")
        # dblScoreDelta = 0  # positive = good for Happy, negative = good for Grumpy
        #
        # for sprBall in self.sprHappyGoal.update_score():
        #     if sprBall.tplColor == const.COLOR_BALL_POS:
        #         dblScoreDelta += const.POINTS_BALL_SCORED
        #     else:
        #         dblScoreDelta -= const.POINTS_BALL_SCORED
        #     sprBall.kill()
        #
        # for sprBall in self.sprGrumpyGoal.update_score():
        #     if sprBall.tplColor == const.COLOR_BALL_POS:
        #         dblScoreDelta -= const.POINTS_BALL_SCORED
        #     else:
        #         dblScoreDelta += const.POINTS_BALL_SCORED
        #     sprBall.kill()
        #
        # dblHappyScore = dblScoreDelta
        # dblGrumpyScore = -dblScoreDelta

        self.on_step_end()

        return self.get_game_state(intTeam=const.TEAM_HAPPY), \
               self.get_reward(int_team=const.TEAM_HAPPY), \
               self.game_is_done(), \
               GameEnv.DebugInfo(
                   self.get_game_state(intTeam=const.TEAM_GRUMPY),
                   self.get_reward(int_team=const.TEAM_GRUMPY)
               )

    def on_step_begin(self):
        for sprSprite in self.grpAllSprites:
            if hasattr(sprSprite, 'on_step_begin'):
                sprSprite.on_step_begin()

    def on_step_end(self):
        for sprSprite in self.grpAllSprites:
            if hasattr(sprSprite, 'on_step_end'):
                sprSprite.on_step_end()

    def on_robot_collision(self, bot1: Robot, bot2: Robot):
        pass  # override in child class

    def get_reward(self, int_team: int = const.TEAM_HAPPY) -> float:
        # DO NOT CHANGE
        # handle scoring by making a child class in RR_ScoreKeepers that overrides this method
        return 0.0

    def get_game_state(self, intTeam=const.TEAM_HAPPY) -> np.ndarray:
        """
        Key pieces we're outputting (not necessarily in this order):
        Center X,Y coords of each robot.
        Center X,Y coords of each robot, prior step.
        Rotation of each robot.
        Rotation of each robot, prior step.
        Center X,Y coords of each ball.
        Center X,Y coords of each ball, prior step.
        :param intTeam: Is this the game state from happy team's perspective? or grumpy's?
        :return: numpy.ndarray
        """
        if intTeam == const.TEAM_HAPPY:  # report happy bots first
            return np.concatenate((
                list(map(self._robot_state, self.lstHappyBots)),
                list(map(self._robot_state, self.lstGrumpyBots)),
                list(map(self._ball_state, self.lstBalls))
            ), axis=None)
        else:  # report grumpy bots first
            return np.concatenate((
                list(map(self._robot_state, self.lstGrumpyBots)),
                list(map(self._robot_state, self.lstHappyBots)),
                list(map(self._ball_state, self.lstBalls))
            ), axis=None)

    def _ball_state(self, sprBall: Ball) -> List[float]:
        return [sprBall.rectDbl.centerx,
                sprBall.rectDbl.centery,
                sprBall.rectDblPriorStep.centerx,
                sprBall.rectDblPriorStep.centery]

    def _robot_state(self, sprRobot: Robot) -> List[float]:
        return [sprRobot.rectDbl.centerx,
                sprRobot.rectDbl.centery,
                sprRobot.rectDbl.rotation,
                sprRobot.rectDblPriorStep.centerx,
                sprRobot.rectDblPriorStep.centery,
                sprRobot.rectDblPriorStep.rotation]

    def game_is_done(self):
        return self.lngStepCount > const.GAME_LENGTH_STEPS or \
               self.sprGrumpyGoal.is_destroyed() or \
               self.sprHappyGoal.is_destroyed() or \
               not bool(self.grpBalls)

    # Add more things here as they come up
    class DebugInfo(dict):
        def __init__(self, adblGrumpyState: np.ndarray, dblGrumpyScore: float):
            super(GameEnv.DebugInfo, self).__init__()
            self.adblGrumpyState = adblGrumpyState  # type: np.ndarray
            self.dblGrumpyScore = dblGrumpyScore  # type: float

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):  # from gym/core.py
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass


class GameEnv_Simple(GameEnv):

    class Direction(Enum):
        FORWARD = 0
        BACKWARD = 1
        LEFT = 2
        RIGHT = 3
        F_L = 4
        F_R = 5
        B_L = 6
        B_R = 7

    _dct_thrust_from_direction = {
        Direction.FORWARD.value: (1,1),
        Direction.BACKWARD.value: (-1,-1),
        Direction.LEFT.value: (-1, 1),
        Direction.RIGHT.value: (1, -1),
        Direction.F_L.value: (0, 1),
        Direction.F_R.value: (1, 0),
        Direction.B_L.value: (-1, 0),
        Direction.B_R.value: (0, -1)
    }

    def __init__(self, lst_starting_config: List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        """Simplified version of the standard GameEnv. Only awards points for chasing positive balls."""
        """ Don't need to override observation_space since _get_game_state() is set correctly. """
        if GameEnv.action_space is None:
            GameEnv.action_space = gym.spaces.Discrete(len(GameEnv_Simple.Direction))
            # todo truly this is "multi-discrete" and not single discrete, so swap this at some point
            # GameEnv.action_space = gym.spaces.MultiDiscrete(
            #     [len(RoboRogby_Discrete.Direction)]*const.NUM_ROBOTS_TOTAL)

        super(GameEnv_Simple, self).__init__(lst_starting_config)

        # for sprRobot in self.lstRobots:
        #     sprRobot.bln_allow_wall_sliding = True

    # def reset(self, bln_randomize_pos :bool = True) -> np.ndarray:
    #     obs = super(GameEnv_Simple, self).reset(bln_randomize_pos)
    #     # for sprRobot in self.lstRobots:
    #     #     sprRobot.bln_allow_wall_sliding = True
    #     return obs

    def step(self, lstArgs: List[Direction]):
        # flatten args into 1-D array, if they're not already (tf passes like this)
        arr_args = np.concatenate(lstArgs, axis=None)
        lst_args_super = []
        if len(arr_args) > const.NUM_ROBOTS_TOTAL:
            raise Exception(f"{len(arr_args)} commands but only {const.NUM_ROBOTS_TOTAL} robots.")
        for i in range(len(arr_args)):
            lst_args_super.append(GameEnv_Simple._dct_thrust_from_direction[arr_args[i]])

        return super(GameEnv_Simple, self).step(lst_args_super)

    """ Simplify the observation space. No 'prior position' elements. """
    def _ball_state(self, sprBall: Ball) -> List[float]:
        return [float(sprBall.rectDbl.centerx),
                float(sprBall.rectDbl.centery)]

    def _robot_state(self, sprRobot: Robot) -> List[float]:
        return [float(sprRobot.rectDbl.centerx),
                float(sprRobot.rectDbl.centery),
                float(sprRobot.rectDbl.rotation)]

class GameEnv_Simple_AngleDistObs(GameEnv_Simple):

    def __init__(self, lst_starting_config: List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        if const.NUM_ROBOTS_HAPPY != 1 or const.NUM_ROBOTS_GRUMPY != 1:
            print("Warning: Observation space/action space is robot-specific (assumes 1 robot per team).")
        super(GameEnv_Simple_AngleDistObs, self).__init__(lst_starting_config)

    def get_game_state(self, intTeam=const.TEAM_HAPPY) -> np.ndarray:
        """
        Move to a more... bayesian-friendly?... observation space (Skyler's original proposed observation-space, iirc).

        For a given robot:
        Bot's current rotation.
        Rotation to positive ball.
        Distance to positive ball.
        Front/back lidar (distance to wall straight out front/straight out back)

        #todo expand this later, if success
        :param intTeam:
        :return:
        """
        if intTeam == const.TEAM_HAPPY:
            return np.asarray(self._robot_state(self.lstHappyBots[0]))
        else:
            return np.asarray(self._robot_state(self.lstGrumpyBots[0]))

    def _robot_state(self, sprRobot: Robot) -> List[float]:
        b = self.lstPosBalls[0]  # type: Ball
        ball_angle = (angle_degrees(sprRobot.rectDbl.center, b.rectDbl.center) + 360) % 360
        ball_dist = abs(distance(sprRobot.rectDbl.center, b.rectDbl.center))

        """ Calc super primitive lidar extending front/back from center of bot """
        top_a, top_b = sprRobot.rectDbl.side(FloatRect.SideType.TOP)
        bottom_a, bottom_b = sprRobot.rectDbl.side(FloatRect.SideType.BOTTOM)
        mid_top = Point(x = (top_a.x + top_b.x) / 2, y = (top_a.y + top_b.y) / 2)
        mid_bot = Point(x = (bottom_a.x + bottom_b.x) / 2, y = (bottom_a.y + bottom_b.y) / 2)

        lst_surfaces = [
            [(0,0), (const.ARENA_WIDTH, 0)],
            [(const.ARENA_WIDTH, 0), (const.ARENA_WIDTH, const.ARENA_HEIGHT)],
            [(const.ARENA_WIDTH, const.ARENA_HEIGHT), (0, const.ARENA_HEIGHT)],
            [(0, const.ARENA_HEIGHT), (0,0)]
        ]

        for other_bot in self.lstRobots:
            if not other_bot is sprRobot:
                for a,b in other_bot.rectDbl.sides:
                    lst_surfaces.append([a,b])

        lst_front = []
        lst_back = []
        for side in lst_surfaces:
            x_i, y_i = get_line_intersection(side, [mid_top, mid_bot])
            if x_i is None or y_i is None:
                pass  # doesn't intersect

            elif mid_bot.x < mid_top.x < x_i or \
                    x_i < mid_top.x < mid_bot.x or \
                    mid_bot.y < mid_top.y < y_i or \
                    y_i < mid_top.y < mid_bot.y:
                """ Viewing this out of the front of the bot """
                lst_front.append(distance(mid_top, (x_i, y_i)))

            elif mid_top.x < mid_bot.x < x_i or \
                    x_i < mid_bot.x < mid_top.x or \
                    mid_top.y < mid_bot.y < y_i or \
                    y_i < mid_bot.y < mid_top.y:
                """ Viewing this out of the back of the bot """
                lst_back.append(distance(mid_bot, (x_i, y_i)))

            elif mid_top.x <= x_i <= mid_bot.x or \
                    mid_bot.x <= x_i <= mid_top.x or \
                    mid_top.y <= y_i <= mid_bot.y or \
                    mid_bot.y <= y_i <= mid_top.y:
                lst_front.append(0)
                lst_back.append(0)
            else:
                raise Exception("This 'am i looking out front or back?' check is dum.")

        lidar_front = min(lst_front)
        lidar_back = min(lst_back)

        return [
            float(sprRobot.dblRotation),
            float(ball_angle),
            float(ball_dist),
            float(lidar_front),
            float(lidar_back)
        ]
