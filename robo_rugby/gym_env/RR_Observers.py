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
    def __init__(self, lst_starting_config: List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
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

    def get_game_state(self, int_team: int = None, obj_robot: Robot = None, obj_ball: Ball = None) -> np.ndarray:
        """ Override in inherited class. """
        return super(AbstractObservers, self).get_game_state(
            int_team=int_team, obj_robot=obj_robot, obj_ball=obj_ball)


# region XYCoord spaces (did not do well with basic "Push ball" in random loc training)

class AllCoords(AbstractObservers):
    """ Location of all elements reported as coords, including robot rotations. """

    def get_game_state(self, int_team: int = None, obj_robot: Robot = None, obj_ball: Ball = None) -> np.ndarray:
        """ todo
        As defined currently, you would need two separate neural nets for Grumpy vs Happy because
        we're not reporting the team or position of the goals and letting the NN infer that
        bottom right = good, top left = bad.

        To properly report the state from Grumpy's perspective, you'd need to flip everything.
        """
        if int_team is None:
            int_team = const.TEAM_HAPPY
        if obj_robot:
            raise NotImplementedError("Robot-specific state output not supported.")

        if int_team == const.TEAM_HAPPY:  # report happy bots first
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

    def get_game_state(self, int_team: int = None, obj_robot: Robot = None, obj_ball: Ball = None) -> np.ndarray:
        if obj_robot:
            return np.asarray(self._robot_state(obj_robot))
        elif int_team == const.TEAM_HAPPY and len(self.lstHappyBots) > 0:
            return np.asarray(self._robot_state(self.lstHappyBots[0]))
        elif int_team == const.TEAM_GRUMPY and len(self.lstGrumpyBots) > 0:
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
        mid_top = Point(x=(top_a.x + top_b.x) / 2, y=(top_a.y + top_b.y) / 2)
        mid_bot = Point(x=(bottom_a.x + bottom_b.x) / 2, y=(bottom_a.y + bottom_b.y) / 2)

        """ Define list of things to check with lidar """
        other_bots = filter(lambda x: not x is sprRobot, self.lstRobots)
        lst_objects = list(map(lambda x: x.rectDbl, other_bots)) + [self.rect_walls]

        lidar_front, lidar_back = TrashyPhysics.two_way_lidar_rect(mid_bot, mid_top, lst_objects)

        return [
            float(sprRobot.dblRotation),
            float(ball_angle),
            float(ball_dist),
            float(lidar_front),
            float(lidar_back)
        ]

class SingleBall_6wayLidar(AbstractObservers):
    """
    For a given robot and ball, the observation is (not necessarily in this order):

        Bot's current rotation.
        Rotation to positive ball.
        Distance to positive ball.
        Rotation to goal the ball should head towards.
        Distance to goal the ball should head towards.
        6 way lidar (front/back/corners)

    If robot/ball aren't defined, it assumes HappyBot[0], PosBall[0].

    TODO delete this version and just use upgraded "v2" version, below (saving for now cuz my saved models
    would be worthless without it.
    """
    def get_game_state(self, int_team: int = None, obj_robot: Robot = None, obj_ball: Ball = None) -> np.ndarray:

        """ Validate """
        if int_team == const.TEAM_HAPPY and obj_robot is None and len(self.lstHappyBots) == 0:
            return None
        if int_team == const.TEAM_GRUMPY and obj_robot is None and len(self.lstGrumpyBots) == 0:
            return None

        if obj_ball is None:
            obj_ball = self.lstPosBalls[0]
        if int_team is not None and obj_robot is not None:
            assert (obj_robot.intTeam == int_team)
        elif int_team is None and obj_robot is None:
            int_team = const.TEAM_HAPPY
            obj_robot = self.lstHappyBots[0]
        elif int_team is None:  # and obj_robot is not None
            int_team = obj_robot.intTeam
        elif obj_robot is None:  # and int_team is not None
            obj_robot = self.lstHappyBots[0] if int_team == const.TEAM_HAPPY else self.lstGrumpyBots[0]

        """ DO LIDAR 
        Front of the robot = Right side of rect (0 rotation)
        Left side of robot = Top side of rect (90) etc.
        """

        # TODO this shit is all fucked up
        # SideType is probably wrong
        # BUT you could still train with it, technically
        # it is reporting all the right information, it's just
        # not in the order that I'm assuming it is here... :(

        front_a, front_b = obj_robot.rectDbl.side(FloatRect.SideType.RIGHT)
        back_a, back_b = obj_robot.rectDbl.side(FloatRect.SideType.LEFT)
        mid_front = Point(x=(front_a.x + front_b.x) / 2, y=(front_a.y + front_b.y) / 2)
        mid_back = Point(x=(back_a.x + back_b.x) / 2, y=(back_a.y + back_b.y) / 2)

        """ Define list of things to check with lidar """
        other_bots = filter(lambda x: not x is obj_robot, self.lstRobots)
        lst_objects = list(map(lambda x: x.rectDbl, other_bots)) + [self.rect_walls]

        lidar_front, lidar_back = TrashyPhysics.two_way_lidar_rect(
            mid_back,
            mid_front,
            lst_objects)

        lidar_front_l, lidar_back_r = TrashyPhysics.two_way_lidar_rect(
            obj_robot.rectDbl.corner(FloatRect.CornerType.BOTTOM_LEFT),
            obj_robot.rectDbl.corner(FloatRect.CornerType.TOP_RIGHT),
            lst_objects
        )

        lidar_front_r, lidar_back_l = TrashyPhysics.two_way_lidar_rect(
            obj_robot.rectDbl.corner(FloatRect.CornerType.TOP_LEFT),
            obj_robot.rectDbl.corner(FloatRect.CornerType.BOTTOM_RIGHT),
            lst_objects
        )

        """ Gather remaining obs """
        ball_angle = angle_degrees(obj_robot.rectDbl.center, obj_ball.rectDbl.center)
        ball_dist = distance(obj_robot.rectDbl.center, obj_ball.rectDbl.center)
        goal_angle = angle_degrees(obj_robot.rectDbl.center, (const.ARENA_WIDTH, const.ARENA_HEIGHT))
        bot_angle = obj_robot.dblRotation

        """
        Standard output is for Happy bot chasing a Positive ball.
        If team is Grumpy, flip it. If ball is Negative, flip it. 
        """
        if (int_team == const.TEAM_HAPPY and obj_ball.is_positive) or \
                (int_team == const.TEAM_GRUMPY and obj_ball.is_negative):

            goal_dist = distance(obj_robot.rectDbl.center, (const.ARENA_WIDTH, const.ARENA_HEIGHT))

        else:  # flip the observation
            goal_dist = distance(obj_robot.rectDbl.center, (0,0))
            ball_angle = (ball_angle + 180) % 360
            goal_angle = (goal_angle + 180) % 360
            bot_angle = (bot_angle + 180) % 360

        # Cap lidar at...
        lidar_cap = 150
        ball_dist = min(ball_dist, lidar_cap)
        goal_dist = min(goal_dist, lidar_cap)
        lidar_front = min(lidar_front, lidar_cap)
        lidar_back = min(lidar_back, lidar_cap)
        lidar_front_l = min(lidar_front_l, lidar_cap)
        lidar_front_r = min(lidar_front_r, lidar_cap)
        lidar_back_r = min(lidar_back_r, lidar_cap)
        lidar_back_l = min(lidar_back_l, lidar_cap)

        return np.asarray([
            float(bot_angle),
            float(ball_angle),
            float(ball_dist),
            float(goal_angle),
            float(goal_dist),
            float(lidar_front),
            float(lidar_front_l),
            float(lidar_front_r),
            float(lidar_back),
            float(lidar_back_l),
            float(lidar_back_r)
        ])

class SingleBall_6wayLidar_v2(AbstractObservers):
    """
    For a given robot and ball, the observation is (not necessarily in this order):

        Bot's current rotation.
        Rotation to positive ball.
        Distance to positive ball.
        Rotation to goal the ball should head towards.
        Distance to goal the ball should head towards.
        6 way lidar (front/back/corners)

    If robot/ball aren't defined, it assumes HappyBot[0], PosBall[0].

    """
    def get_game_state(self, int_team: int = None, obj_robot: Robot = None, obj_ball: Ball = None) -> np.ndarray:

        """ Validate """
        if int_team == const.TEAM_HAPPY and obj_robot is None and len(self.lstHappyBots) == 0:
            return None
        if int_team == const.TEAM_GRUMPY and obj_robot is None and len(self.lstGrumpyBots) == 0:
            return None

        if obj_ball is None:
            obj_ball = self.lstPosBalls[0]
        if int_team is not None and obj_robot is not None:
            assert (obj_robot.intTeam == int_team)
        elif int_team is None and obj_robot is None:
            int_team = const.TEAM_HAPPY
            obj_robot = self.lstHappyBots[0]
        elif int_team is None:  # and obj_robot is not None
            int_team = obj_robot.intTeam
        elif obj_robot is None:  # and int_team is not None
            obj_robot = self.lstHappyBots[0] if int_team == const.TEAM_HAPPY else self.lstGrumpyBots[0]

        """ DO LIDAR 
        Front of the robot = Right side of rect (0 rotation)
        Left side of robot = Top side of rect (90) etc.
        """

        # TODO this shit is all fucked up
        # SideType is probably wrong
        # BUT you could still train with it, technically
        # it is reporting all the right information, it's just
        # not in the order that I'm assuming it is here... :(

        front_a, front_b = obj_robot.rectDbl.side(FloatRect.SideType.RIGHT)
        back_a, back_b = obj_robot.rectDbl.side(FloatRect.SideType.LEFT)
        mid_front = Point(x=(front_a.x + front_b.x) / 2, y=(front_a.y + front_b.y) / 2)
        mid_back = Point(x=(back_a.x + back_b.x) / 2, y=(back_a.y + back_b.y) / 2)

        """ Define list of things to check with lidar """
        other_bots = filter(lambda x: not x is obj_robot, self.lstRobots)
        lst_objects = list(map(lambda x: x.rectDbl, other_bots)) + [self.rect_walls]

        lidar_front, lidar_back = TrashyPhysics.two_way_lidar_rect(
            mid_back,
            mid_front,
            lst_objects)

        lidar_front_l, lidar_back_r = TrashyPhysics.two_way_lidar_rect(
            obj_robot.rectDbl.corner(FloatRect.CornerType.BOTTOM_LEFT),
            obj_robot.rectDbl.corner(FloatRect.CornerType.TOP_RIGHT),
            lst_objects
        )

        lidar_front_r, lidar_back_l = TrashyPhysics.two_way_lidar_rect(
            obj_robot.rectDbl.corner(FloatRect.CornerType.TOP_LEFT),
            obj_robot.rectDbl.corner(FloatRect.CornerType.BOTTOM_RIGHT),
            lst_objects
        )

        # Cap lidar at...
        lidar_cap = 150
        lidar_front = min(lidar_front, lidar_cap)
        lidar_back = min(lidar_back, lidar_cap)
        lidar_front_l = min(lidar_front_l, lidar_cap)
        lidar_front_r = min(lidar_front_r, lidar_cap)
        lidar_back_r = min(lidar_back_r, lidar_cap)
        lidar_back_l = min(lidar_back_l, lidar_cap)

        """ Gather remaining obs """
        ball_angle = angle_degrees(obj_robot.rectDbl.center, obj_ball.rectDbl.center)
        ball_dist = distance(obj_robot.rectDbl.center, obj_ball.rectDbl.center)
        ball_dist = min(ball_dist, lidar_cap)
        goal_angle = angle_degrees(obj_robot.rectDbl.center, (const.ARENA_WIDTH, const.ARENA_HEIGHT))
        bot_angle = obj_robot.dblRotation

        goal_dist_cap = max(const.GOAL_WIDTH, const.GOAL_HEIGHT) + lidar_cap
        bad_goal_dist = distance(obj_robot.rectDbl.center, (0,0))
        good_goal_dist = distance(obj_robot.rectDbl.center, (const.ARENA_WIDTH, const.ARENA_HEIGHT))
        if good_goal_dist <= bad_goal_dist:
            goal_dist = min(good_goal_dist, goal_dist_cap)
        else:
            goal_dist = -1 * min(bad_goal_dist, goal_dist_cap)

        """
        Standard output is for Happy bot chasing a Positive ball.
        If team is Grumpy, flip it. If ball is Negative, flip it. 
        """
        if (int_team == const.TEAM_HAPPY and obj_ball.is_negative) or \
                (int_team == const.TEAM_GRUMPY and obj_ball.is_positive):
            # flip the observation
            goal_dist *= -1
            ball_angle = (ball_angle + 180) % 360
            goal_angle = (goal_angle + 180) % 360
            bot_angle = (bot_angle + 180) % 360

        return np.asarray([
            float(bot_angle),
            float(ball_angle),
            float(ball_dist),
            float(goal_angle),
            float(goal_dist),
            float(lidar_front),
            float(lidar_front_l),
            float(lidar_front_r),
            float(lidar_back),
            float(lidar_back_l),
            float(lidar_back_r)
        ])



