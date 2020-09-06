from __future__ import absolute_import, division, print_function
from typing import List, Tuple
import MyUtils
from MyUtils import Stage
import gym
from gym.utils import seeding
import pygame
import math
import random
import robo_rugby.gym_env.RR_Constants as const
import numpy as np
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

def get_tf_wrapped_robo_rugby_env():
    """Wraps given gym environment with TF Agent's GymWrapper.
      Note that by default a TimeLimit wrapper is used to limit episode lengths
      to the default benchmarks defined by the registered environments.
      Args:
        gym_env: An instance of OpenAI gym environment.
        discount: Discount to use for the environment.
        max_episode_steps: Used to create a TimeLimitWrapper. No limit is applied
          if set to None or 0. Usually set to `gym_spec.max_episode_steps` in `load.
        gym_env_wrappers: Iterable with references to wrapper classes to use
          directly on the gym environment.
        time_limit_wrapper: Wrapper that accepts (env, max_episode_steps) params to
          enforce a TimeLimit. Usuaully this should be left as the default,
          wrappers.TimeLimit.
        env_wrappers: Iterable with references to wrapper classes to use on the
          gym_wrapped environment.
        spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
          default dtype for the tensors. An easy way how to configure a custom
          mapping through Gin is to define a gin-configurable function that returns
          desired mapping and call it in your Gin config file, for example:
          `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.
        auto_reset: If True (default), reset the environment automatically after a
          terminal state is reached.
        render_kwargs: Optional `dict` of keywoard arguments for rendering.
      Returns:
        A PyEnvironment instance.
      """
    from tf_agents.environments import suite_gym
    gym_spec = gym.spec("RoboRugby-v0")
    gym_env = gym_spec.make()
    return suite_gym.wrap_env(
        gym_env,
        discount=1.0,  # discount TODO research that more
        max_episode_steps=gym_spec.max_episode_steps,
        auto_reset=False)
    # gym_env_wrappers=gym_env_wrappers,
    # time_limit_wrapper=wrappers.TimeLimit,  # default
    # env_wrappers=env_wrappers,
    # spec_dtype_map=spec_dtype_map,


class GameEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second':const.FRAMERATE
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

    def __init__(self, lst_starting_config:List[Tuple[float,float]] = CONFIG_STANDARD):
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
            self.lstRobots[i] = Robot(const.TEAM_HAPPY, (0,0))
            self.grpRobots.add(self.lstRobots[i])
            self.grpAllSprites.add(self.lstRobots[i])
        for i in range(const.NUM_ROBOTS_HAPPY, lngTotalRobots):
            self.lstRobots[i] = Robot(const.TEAM_GRUMPY, (0,0))
            self.grpRobots.add(self.lstRobots[i])
            self.grpAllSprites.add(self.lstRobots[i])

        Stage("Give it some balls!")
        self.grpBalls = pygame.sprite.Group()
        lngTotalBalls = const.NUM_BALL_POS + const.NUM_BALL_NEG
        self.lstBalls = [None] * lngTotalBalls  # type: List[Ball]
        for i in range(const.NUM_BALL_POS):
            self.lstBalls[i] = Ball(const.COLOR_BALL_POS, (0,0))
            self.grpBalls.add(self.lstBalls[i])
            self.grpAllSprites.add(self.lstBalls[i])          
        for i in range(const.NUM_BALL_POS, lngTotalBalls):
            self.lstBalls[i] = Ball(const.COLOR_BALL_NEG, (0,0))
            self.grpBalls.add(self.lstBalls[i])
            self.grpAllSprites.add(self.lstBalls[i])
            
        Stage("Set the starting positions")
        if lst_starting_config is None:
            self._lst_starting_positions = self._set_random_positions()
        else:
            self._lst_starting_positions = lst_starting_config
            self._set_starting_positions()
            
        self.dblBallDistSum = self._get_ball_distance_sum()

        # Minitaur gym is a good example of similar inputs/outputs
        # Velocity/position/rotation can never be > bounds of the arena... barring crazy constants
        if GameEnv.observation_space is None:  # shared variable from base class, gym.Env()
            arrState = self._get_game_state()
            dblObservationHigh = max(const.ARENA_WIDTH, const.ARENA_HEIGHT, 360)
            dblObservationLow = dblObservationHigh * -1
            GameEnv.observation_space = gym.spaces.Box(
                dblObservationLow, dblObservationHigh, dtype=np.float32, shape=arrState.shape)

        if GameEnv.action_space is None:  # shared variable from base class, gym.Env()
            alngActions = np.array([1]*len(self.lstRobots)*2)
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
        if(len(self._lst_starting_positions) != 2):
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
                spr_robot.rectDbl.centerx = random.randint(const.ARENA_WIDTH / 2 + const.ROBOT_WIDTH * 2,
                                                   const.ARENA_WIDTH - const.ROBOT_WIDTH * 2)
                spr_robot.rectDbl.centery = random.randint(const.ARENA_HEIGHT / 2 + const.ROBOT_LENGTH * 2,
                                                   const.ARENA_HEIGHT - const.ROBOT_LENGTH * 2)
                spr_robot.dblRotation = random.randint(0,360)
                if len(pygame.sprite.spritecollide(spr_robot, self.grpRobots, False)) <= 1:
                    break  # always collides with self

        for spr_robot in self.lstGrumpyBots:
            while True:
                # Grumpy team starts in the top-left quad with 2-robots padding
                spr_robot.rectDbl.centerx = random.randint(const.ROBOT_WIDTH * 2,
                                                   const.ARENA_WIDTH / 2 - const.ROBOT_WIDTH * 2)
                spr_robot.rectDbl.centery = random.randint(const.ROBOT_LENGTH * 2,
                                                   const.ARENA_HEIGHT / 2 - const.ROBOT_LENGTH * 2)
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

    def reset(self, bln_randomize_pos :bool = False) -> np.ndarray:
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
        self.dblBallDistSum = self._get_ball_distance_sum()

        return self._get_game_state()

    def render(self, mode='human'):

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
        dblGrumpyScore = -1 * const.POINTS_TIME_PENALTY
        dblHappyScore = -1 * const.POINTS_TIME_PENALTY
        setNaughtyRobots = set()

        Stage("Prep ze sprites!")
        for sprSprite in self.grpAllSprites:
            if hasattr(sprSprite, 'on_step_begin'):
                sprSprite.on_step_begin()

        Stage("Activate robot engines!")
        # flatten args into 1-D array, if they're not already (tf passes like this)
        arr_args = np.concatenate(lstArgs, axis=None)
        if len(arr_args) > const.NUM_ROBOTS_TOTAL * 2:
            raise Exception(f"{len(arr_args)} commands but only {const.NUM_ROBOTS_TOTAL*2} robot engines.")
        for i in range(0, len(arr_args), 2):
            self.lstRobots[int(i/2)].set_thrust(arr_args[i], arr_args[i+1])

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

        return self._get_game_state(intTeam=const.TEAM_HAPPY), \
               dblHappyScore, \
               self.game_is_done(), \
               GameEnv.DebugInfo(
                   self._get_game_state(intTeam=const.TEAM_GRUMPY),
                   dblGrumpyScore
               )

    def _get_game_state(self, intTeam=const.TEAM_HAPPY) -> np.ndarray:
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

    def _ball_state(self, sprBall :Ball) -> List[float]:
        return [sprBall.rectDbl.centerx,
                sprBall.rectDbl.centery,
                sprBall.rectDblPriorStep.centerx,
                sprBall.rectDblPriorStep.centery]

    def _robot_state(self, sprRobot :Robot) -> List[float]:
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
    
    def _get_ball_distance_sum(self) -> float:
        """ Distance meaning 'Distance from Grumpy's goal'"""
        def _dist(sprBall:Ball)->float:
            return math.pow(sprBall.rectDbl.centerx**2 + sprBall.rectDbl.centery**2, .5)
        
        # Higher score the farther away positive balls get from Grumpy's goal (and therefore closer to Happy's goal)
        # Lower score the farther away negative balls get from Grumpy's goal (and therefore closer to Happy's goal)
        return sum(map(_dist, self.lstPosBalls)) - sum(map(_dist, self.lstNegBalls))

    # Add more things here as they come up
    class DebugInfo:
        def __init__(self, adblGrumpyState:np.ndarray, dblGrumpyScore:float):
            self.adblGrumpyState = adblGrumpyState  #type: np.ndarray
            self.dblGrumpyScore = dblGrumpyScore  #type: float

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):  # from gym/core.py
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass







