from robo_rugby.gym_env.RR_Robot import Robot
from . import RR_TrashyPhysics as TrashyPhysics
from robo_rugby.gym_env.RR_EnvBase import GameEnv, GameEnv_Simple
from . import RR_Constants as const
from robo_rugby.gym_env.RR_Ball import Ball
from MyUtils import distance, angle_degrees, FloatRect, Point, get_line_intersection
from typing import Set, List, Tuple
import numpy as np
import gym

""" 
Decorator design pattern for observation systems (for easily swapping in/out)
"""
class AbstractObservers(GameEnv):
    def __init__(self, lst_starting_config:List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        super(AbstractObservers, self).__init__(lst_starting_config)

        # Observers overwrite each other, so only one is expected. Bom otherwise.
        observer_count = 0
        for cls in type(self).__bases__:
            if issubclass(cls, AbstractObservers):
                observer_count += 1
            if observer_count > 1:
                raise Exception(f"Multiple observers defined for class '{type(self).__name__}'.")

        self.set_observation_space()

    def set_observation_space(self):
        """ Override if needed (but I'm assuming most obs spaces will adhere to this) """
        if AbstractObservers.observation_space is None:  # shared variable from base class, gym.Env()
            arrState = self.get_game_state()
            dblObservationHigh = max(const.ARENA_WIDTH, const.ARENA_HEIGHT, 360)
            dblObservationLow = dblObservationHigh * -1
            AbstractObservers.observation_space = gym.spaces.Box(
                dblObservationLow, dblObservationHigh, dtype=np.float32, shape=arrState.shape)

    def get_game_state(self, intTeam=const.TEAM_HAPPY, obj_robot=None) -> np.ndarray:
        """ Override in inherited class. """
        return super(AbstractObservers, self).get_game_state(intTeam=intTeam)


# region XYCoord spaces (did not do well with basic "Push ball" in random loc training)

class AllCoords(AbstractObservers):
    """ Location of all elements reported as coords, including robot rotations. """

    def get_game_state(self, intTeam=const.TEAM_HAPPY, obj_robot=None) -> np.ndarray:
        """ todo
        As defined currently, you would need two separate neural nets for Grumpy vs Happy because
        we're not reporting the team or position of the goals and letting the NN infer that
        bottom right = good, top left = bad.

        To properly report the state from Grumpy's perspective, you'd need to flip everything.
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
        return [float(sprBall.rectDbl.centerx),
                float(sprBall.rectDbl.centery)]

    def _robot_state(self, sprRobot: Robot) -> List[float]:
        return [float(sprRobot.rectDbl.centerx),
                float(sprRobot.rectDbl.centery),
                float(sprRobot.rectDbl.rotation)]


class AllCoords_WithPrior(AllCoords):
    """ Location of all elements reported as coords + prior locations. Also includes robot rotations.

    Key pieces we're outputting (not necessarily in this order):
        Center X,Y coords of each robot.
        Center X,Y coords of each robot, prior step.
        Rotation of each robot.
        Rotation of each robot, prior step.
        Center X,Y coords of each ball.
        Center X,Y coords of each ball, prior step.
    """

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

# endregion


class PosBall_BasicLidar(AbstractObservers):
    """
    For a given robot, the observation is:

        Bot's current rotation.
        Rotation to positive ball.
        Distance to positive ball.
        Front/back lidar (distance to wall straight out front/straight out back)

    Assumes 1 positive ball.
    """

    def __init__(self, lst_starting_config: List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        if const.NUM_ROBOTS_HAPPY != 1 or const.NUM_ROBOTS_GRUMPY != 1:
            print("Warning: Observation space/action space is robot-specific (assumes 1 robot per team).")
        super(PosBall_BasicLidar, self).__init__(lst_starting_config)

    def get_game_state(self, intTeam=const.TEAM_HAPPY, obj_robot=None) -> np.ndarray:
        if obj_robot:
            return np.asarray(self._robot_state(obj_robot))
        elif intTeam == const.TEAM_HAPPY and len(self.lstHappyBots) > 0:
            return np.asarray(self._robot_state(self.lstHappyBots[0]))
        elif intTeam == const.TEAM_GRUMPY and len(self.lstGrumpyBots) > 0:
            return np.asarray(self._robot_state(self.lstGrumpyBots[0]))
        else:
            return None

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
