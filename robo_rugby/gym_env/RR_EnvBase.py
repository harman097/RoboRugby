from __future__ import absolute_import, division, print_function
from typing import List, Tuple, Set
import MyUtils
from MyUtils import Stage, Point, FloatRect, distance, angle_degrees, angle_radians, Div0, get_slope_yint, get_line_intersection
import gym
from gym.utils import seeding
import pygame
import math
import random
from . import RR_Constants as const
import numpy as np
from enum import Enum
from PIL import Image

Stage("Initialize pygame")
pygame.init()
mScreen = pygame.display.set_mode((const.ARENA_WIDTH + const.DASHBOARD_WIDTH, const.ARENA_HEIGHT))

from robo_rugby.gym_env.RR_Ball import Ball
from robo_rugby.gym_env.RR_Robot import Robot
from robo_rugby.gym_env.RR_Goal import Goal
from robo_rugby.gym_env.RR_Dashboard import Dashboard
from . import RR_TrashyPhysics as TrashyPhysics

MyUtils.PRINT_STAGE = False  # Disable stage spam

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
        self.rect_walls = FloatRect(0, const.ARENA_WIDTH, 0, const.ARENA_HEIGHT)

        Stage("Menu for you, sir")
        self.dashboard = Dashboard()

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
                        
        self.dashboard.update(mScreen, 
                              {"Score, Happy": self.sprHappyGoal.get_score(), 
                               "Score, Grumpy": self.sprGrumpyGoal.get_score()
                               })
        
        
        

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

        self.lngStepCount += 1
        self.on_step_begin()

        """ Activate robot engines! """
        # flatten args into 1-D array, if they're not already (tf passes like this)
        arr_args = np.concatenate(lstArgs, axis=None)
        if len(arr_args) > const.NUM_ROBOTS_TOTAL * 2:
            raise Exception(f"{len(arr_args)} commands but only {const.NUM_ROBOTS_TOTAL * 2} robot engines.")
        for i in range(0, len(arr_args), 2):
            self.lstRobots[int(i / 2)].set_thrust(arr_args[i], arr_args[i + 1])

        for _ in range(const.MOVES_PER_FRAME):

            set_bots_that_moved = set(self.lstRobots)
            set_balls_that_moved = set(self.lstBalls)
            
            self.on_frame_begin()
            self._move_bots()
            self._resolve_bot_collisions(set_bots_that_moved)
            self._push_balls()
            self._roll_balls()
            if not self._resolve_ball_collisions():
                self._undo_naughty_movement(set_balls_that_moved, set_bots_that_moved)
            self.on_frame_end()

        self.on_step_end()

        return self.get_game_state(), \
               self.get_reward(int_team=const.TEAM_HAPPY), \
               self.game_is_done(), \
               GameEnv.DebugInfo(
                   self.get_game_state(int_team=const.TEAM_GRUMPY),
                   self.get_reward(int_team=const.TEAM_GRUMPY)
               )

    def _move_bots(self):
        for robot in self.lstRobots:
            robot.move()

    def _resolve_bot_collisions(self, set_bots_that_moved:Set[Robot]):

        lst_bot_collisions = TrashyPhysics.collision_pairs_self(
                self.grpRobots, fncCollided=TrashyPhysics.robots_collided)
        lng_attempts = 0
        lng_attempt_limit = len(self.lstRobots)  # Max possible needed (conga line and the first guy doesn't move)

        while len(lst_bot_collisions) > 0:
            lng_attempts += 1
            if lng_attempts > lng_attempt_limit:
                raise Exception("UNABLE TO RESOLVE BOT/BOT COLLISIONS")

            for bot1, bot2 in lst_bot_collisions:
                self.on_robot_collision(bot1, bot2)

                """ Undo move for each robot, if possible """
                bln_stuck = True
                robot :Robot
                for robot in [bot1, bot2]:
                    if robot in set_bots_that_moved:
                        set_bots_that_moved.remove(robot)
                        if not robot.undo_move():
                            raise Exception(f"UNABLE TO UNDO MOVE FOR ROBOT: {robot}")
                        bln_stuck = False
                if bln_stuck:
                    raise Exception("""ROBOTS STUCK FROM PRIOR FRAME.
                    Collisions should be fully resolved at the end of each frame.
                    """)

            lst_bot_collisions = TrashyPhysics.collision_pairs_self(
                self.grpRobots, fncCollided=TrashyPhysics.robots_collided)

    def _push_balls(self):
        for sprBall, sprRobot in TrashyPhysics.collision_pairs(
                self.grpBalls, self.grpRobots, fncCollided=TrashyPhysics.ball_robot_collided):
            TrashyPhysics.apply_force_to_ball(sprRobot, sprBall)
            TrashyPhysics.bounce_ball_off_bot(sprRobot, sprBall)

    def _roll_balls(self):
        for ball in self.lstBalls:
            ball.move()

    def _resolve_ball_collisions(self) -> bool:
        """ Rough pseudo
        (6) TryResolve loop
            Naughty = True
            While Naughty:
                Naughty = False
                (6a) Resolve ball/ball collisions (bounce)
                (6b) Resolve ball/bot collisions (bounce)
                (6c) Resolve ball/wall collisions (bounce)
                (6d) Resolve ball/bumper collisions (bounce)
                Naughty = True so long as there are any
                TryResolveLoopLimit? Break

        :param set_balls_that_moved:
        :param set_bots_that_moved:
        :return: Bool: Were all collisions resolved?
        """

        bln_naughty = True
        lng_naughty_loop_count = 0
        lng_naughty_loop_limit = 10
        while bln_naughty:
            lng_naughty_loop_count += 1
            if lng_naughty_loop_count > lng_naughty_loop_limit:
                return False
            bln_naughty = False

            """ Ball vs Ball """
            for sprBall1, sprBall2 in TrashyPhysics.collision_pairs_self(
                    self.grpBalls, fncCollided=TrashyPhysics.balls_collided):
                bln_naughty = True
                TrashyPhysics.bounce_balls(sprBall1, sprBall2)

            """ Ball vs Bot """
            for sprBall, sprRobot in TrashyPhysics.collision_pairs(
                    self.grpBalls, self.grpRobots,
                    fncCollided=TrashyPhysics.ball_robot_collided):
                bln_naughty = True
                TrashyPhysics.bounce_ball_off_bot(sprRobot, sprBall)

            """ Ball vs Wall """
            for ball in filter(lambda x: TrashyPhysics.collided_wall(x), self.lstBalls):
                bln_naughty = True
                TrashyPhysics.bounce_ball_off_wall(ball)

            """ Ball vs Bumper """
            # todo

        return True

    def _undo_naughty_movement(self, set_balls_that_moved :Set[Ball], set_bots_that_moved :Set[Robot]):
        """ Rough pseudo
        (7) UndoNaughty loop
            While Naughty:
                Naughty = False
                (7a) Undo moves for ball/ball collisions (remove from set of _ who moved)
                (7b) Undo moves for ball/bot collisions (remove from set of _ who moved)
                (7c) Undo moves for ball/wall collisions (bounce) (remove from set of _ who moved)
                (7d) Undo moves for ball/bumper collisions (bounce) (remove from set of _ who moved)
                Naughty = True so long as there are any
                UndoNaughtyLoopLimit? BOM
        :param set_balls_that_moved:
        :param set_bots_that_moved:
        :return:
        """
        ball :Ball
        bot :Robot
        bln_naughty = True
        lng_naughty_loop_count = 0
        lng_naughty_loop_limit = len(self.lstBalls) + len(self.lstRobots)  # worst case scenario
        while bln_naughty:
            lng_naughty_loop_count += 1
            if lng_naughty_loop_count > lng_naughty_loop_limit:
                if const.GAME_MODE:
                    print("WARNING: UNABLE TO RESOLVE ALL COLLISIONS FOR FRAME.")
                else:
                    raise Exception("UNABLE TO RESOLVE ALL COLLISIONS FOR FRAME")
            set_naughty_bots = set()
            set_naughty_balls = set()

            """ Ball vs Ball """
            for ball1, ball2 in TrashyPhysics.collision_pairs_self(
                    self.grpBalls, fncCollided=TrashyPhysics.balls_collided):
                set_naughty_balls.add(ball1)
                set_naughty_balls.add(ball2)

            """ Ball vs Bot """
            for ball, bot in TrashyPhysics.collision_pairs(
                    self.grpBalls, self.grpRobots,
                    fncCollided=TrashyPhysics.ball_robot_collided):
                set_naughty_balls.add(ball)
                set_naughty_bots.add(bot)

            """ Ball vs Wall """
            for ball in filter(lambda x: TrashyPhysics.collided_wall(x), self.lstBalls):
                set_naughty_balls.add(ball)

            """ Ball vs Bumper """
            # todo

            """ Undo movement. """
            bln_naughty = len(set_naughty_bots) + len(set_naughty_balls) > 0
            for bot in set_bots_that_moved & set_naughty_bots:
                set_bots_that_moved.remove(bot)
                if not bot.undo_move():
                    print(f"UNABLE TO UNDO MOVE FOR BOT: {bot}")
            for ball in set_balls_that_moved & set_naughty_balls:
                set_balls_that_moved.remove(ball)
                if not ball.undo_move():
                    print(f"UNABLE TO UNDO MOVE FOR BALL: {ball}")



    def __old_step(self):

        setBallsInGoal = set()

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

        return self.get_game_state(), \
               self.get_reward(int_team=const.TEAM_HAPPY), \
               self.game_is_done(), \
               GameEnv.DebugInfo(
                   self.get_game_state(int_team=const.TEAM_GRUMPY),
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

    def on_frame_begin(self):
        for sprSprite in self.grpAllSprites:
            if hasattr(sprSprite, 'on_frame_begin'):
                sprSprite.on_frame_begin()

    def on_frame_end(self):
        for sprSprite in self.grpAllSprites:
            if hasattr(sprSprite, 'on_frame_end'):
                sprSprite.on_frame_end()

    def on_robot_collision(self, bot1: Robot, bot2: Robot):
        pass  # override in child class

    def get_reward(self, int_team: int = const.TEAM_HAPPY) -> float:
        # DO NOT CHANGE
        # handle scoring by making a child class in RR_ScoreKeepers that overrides this method
        return 0.0

    def get_game_state(self, int_team: int = None, obj_robot: Robot = None, obj_ball: Ball = None) -> np.ndarray:
        # DO NOT CHANGE
        # handle definition of the game state (observations) by making a child class in RR_Observers that overrides this method
        return None

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
    """ Simplified version of GameEnv with a discrete action space. """

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

    @staticmethod
    def thrust_from_direction(direction: int) -> Tuple[float, float]:
        return GameEnv_Simple._dct_thrust_from_direction[direction]

    def __init__(self, lst_starting_config: List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        if GameEnv.action_space is None:
            GameEnv.action_space = gym.spaces.Discrete(len(GameEnv_Simple.Direction))
            # todo truly this is "multi-discrete" and not single discrete, so swap this at some point
            # GameEnv.action_space = gym.spaces.MultiDiscrete(
            #     [len(RoboRogby_Discrete.Direction)]*const.NUM_ROBOTS_TOTAL)

        super(GameEnv_Simple, self).__init__(lst_starting_config)

    def step(self, lstArgs: List[Direction]):
        # flatten args into 1-D array, if they're not already (tf passes like this)
        arr_args = np.concatenate(lstArgs, axis=None)
        lst_args_super = []
        if len(arr_args) > const.NUM_ROBOTS_TOTAL:
            raise Exception(f"{len(arr_args)} commands but only {const.NUM_ROBOTS_TOTAL} robots.")
        for i in range(len(arr_args)):
            lst_args_super.append(GameEnv_Simple.thrust_from_direction(arr_args[i]))

        return super(GameEnv_Simple, self).step(lst_args_super)
