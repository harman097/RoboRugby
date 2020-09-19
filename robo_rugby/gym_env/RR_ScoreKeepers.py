from robo_rugby.gym_env.RR_Robot import Robot
from . import RR_TrashyPhysics as TrashyPhysics
from robo_rugby.gym_env.RR_EnvBase import GameEnv
from . import RR_Constants as const
from MyUtils import distance
from typing import Set, List, Tuple

""" 
Scoring systems were getting a little messy and I find myself 
enabling/disabling certain ones frequently, so I'm moving this
to a Decorator design pattern* for calculating score.

*Technically supposed to wrap an instance and store a ref to the
"decorated instance", but with Python super() calls the end
result is basically the same and the normal way might not work
with how gym.make_env() instantiates (just gets passed a type
to instantiate so... might not be an opportunity to "decorate").
"""
class AbstractScoreKeeper(GameEnv):
    def __init__(self, lst_starting_config:List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        super(AbstractScoreKeeper, self).__init__(lst_starting_config)        
        self.reward_happy = 0.0
        self.reward_grumpy = 0.0        
        self.max_reward = float("inf")
        self.min_reward = float("-inf")

    def get_reward(self, int_team: int = const.TEAM_HAPPY) -> float:
        return self.reward_happy if int_team == const.TEAM_HAPPY else self.reward_grumpy

    def on_step_begin(self):
        super(AbstractScoreKeeper, self).on_step_begin()
        self.reward_happy = 0.0
        self.reward_grumpy = 0.0

    def on_step_end(self):
        """ Adjust reward here. Override in inherited class. """
        super(AbstractScoreKeeper, self).on_step_end()

    """ ADD MORE EVENTS HERE, AS NEEDED. """

    def on_robot_collision(self, bot1 :Robot, bot2 :Robot):
        """ Adjust reward here. Override in inherited class. """
        super(AbstractScoreKeeper, self).on_robot_collision(bot1, bot2)


class ChasePosBall(AbstractScoreKeeper):
    """
    Super basic. Just chase the freaking good balls. Don't care what you do with them.

    Mainly intended for 1-ball games, but could be useful long term with high enough gamma.
    """

    def on_step_end(self):
        super(ChasePosBall, self).on_step_end()

        for sprRobot in self.lstHappyBots:
            for sprBall in self.lstPosBalls:
                dist_now = distance(sprRobot.rectDbl.center, sprBall.rectDbl.center)
                dist_prior = distance(sprRobot.rectDblPriorStep.center, sprBall.rectDbl.center)
                self.reward_happy += (dist_prior - dist_now) * const.POINTS_ROBOT_TRAVEL_MULT

        for sprRobot in self.lstGrumpyBots:
            for sprBall in self.lstPosBalls:
                dist_now = distance(sprRobot.rectDbl.center, sprBall.rectDbl.center)
                dist_prior = distance(sprRobot.rectDblPriorStep.center, sprBall.rectDbl.center)
                self.reward_grumpy += (dist_prior - dist_now) * const.POINTS_ROBOT_TRAVEL_MULT


class DontDriveInGoals(AbstractScoreKeeper):
    """ If you drive in the goal, you get penalized. Thems the rules. """

    def on_step_end(self):
        super(DontDriveInGoals, self).on_step_end()

        for sprRobot in self.lstRobots:
            if TrashyPhysics.robot_in_goal(sprRobot, self.sprHappyGoal) or \
                    TrashyPhysics.robot_in_goal(sprRobot, self.sprGrumpyGoal):
                if sprRobot.intTeam == const.TEAM_HAPPY:
                    self.reward_happy -= const.POINTS_ROBOT_IN_GOAL_PENALTY
                else:
                    self.reward_grumpy -= const.POINTS_ROBOT_IN_GOAL_PENALTY


class KeepMovingGuys(AbstractScoreKeeper):
    """ Don't move? Get penalized. Lazy robots... """

    def on_step_end(self):
        super(KeepMovingGuys, self).on_step_end()
        for sprRobot in self.lstRobots:
            if sprRobot.rectDbl.center == sprRobot.rectDblPriorStep.center and \
                    sprRobot.rectDbl.rotation == sprRobot.rectDblPriorStep.rotation:

                if sprRobot.intTeam == const.TEAM_HAPPY:
                    self.reward_happy -= const.POINTS_NO_MOVE_PENALTY
                else:
                    self.reward_grumpy -= const.POINTS_NO_MOVE_PENALTY


class BaseDestruction(AbstractScoreKeeper):
    """ If a base is destroyed (3 neg balls) there's a LOT of points we need to dish out. """

    def on_step_end(self):
        super(BaseDestruction, self).on_step_end()
        if self.sprHappyGoal.is_destroyed():
            self.reward_happy += const.POINTS_GOAL_DESTROYED
            self.reward_grumpy -= const.POINTS_GOAL_DESTROYED
        elif self.sprGrumpyGoal.is_destroyed():
            self.reward_happy += const.POINTS_GOAL_DESTROYED
            self.reward_grumpy -= const.POINTS_GOAL_DESTROYED


class NaughtyBots(AbstractScoreKeeper):
    """ This isn't American bumper cars, mmk? No smashing. Wait your damn turn. """

    def __init__(self, lst_starting_config:List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        super(NaughtyBots, self).__init__(lst_starting_config)
        self.set_naughty_bots = set()  # type: Set[Robot]

    def on_step_begin(self):
        super(NaughtyBots, self).on_step_begin()
        self.set_naughty_bots.clear()

    def on_robot_collision(self, bot1 :Robot, bot2 :Robot):
        super(NaughtyBots, self).on_robot_collision(bot1, bot2)
        if bot1.lngLThrust != 0 or bot1.lngRThrust != 0:
            self.set_naughty_bots.add(bot1)
        if bot2.lngLThrust != 0 or bot2.lngRThrust != 0:
            self.set_naughty_bots.add(bot2)

    def on_step_end(self):
        for sprNaughtyBot in self.set_naughty_bots:
            if sprNaughtyBot.intTeam == const.TEAM_HAPPY:
                self.reward_happy -= const.POINTS_ROBOT_CRASH_PENALTY
            else:
                self.reward_grumpy -= const.POINTS_ROBOT_CRASH_PENALTY


class PushPosBallsToGoal(AbstractScoreKeeper):
    """ Positive points for pushing towards your goal. Neg points otherwise. 0-sum. """

    def __init__(self, lst_starting_config:List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        super(PushPosBallsToGoal, self).__init__(lst_starting_config)
        self.ball_dist_sum = 0.0

    def on_step_begin(self):
        super(PushPosBallsToGoal, self).on_step_begin()
        self.ball_dist_sum = self._calc_ball_dist_sum()

    def on_step_end(self):
        super(PushPosBallsToGoal, self).on_step_end()
        ball_dist_delta = self._calc_ball_dist_sum() - self.ball_dist_sum
        self.reward_happy += ball_dist_delta * const.POINTS_BALL_TRAVEL_MULT
        self.reward_grumpy -= ball_dist_delta * const.POINTS_BALL_TRAVEL_MULT

    def _calc_ball_dist_sum(self):
        # Grumpy's goal is in the 0,0 corner, therefore higher distance is better for Happy team
        return sum(map(lambda x: distance((0,0), x.rectDbl.center), self.lstPosBalls))


class PushNegBallsFromGoal(AbstractScoreKeeper):
    """ Positive points for pushing towards enemy goal. Neg points otherwise. 0-sum. """

    def __init__(self, lst_starting_config:List[Tuple[float, float]] = GameEnv.CONFIG_RANDOM):
        super(PushNegBallsFromGoal, self).__init__(lst_starting_config)
        self.ball_dist_sum = 0.0

    def on_step_begin(self):
        super(PushNegBallsFromGoal, self).on_step_begin()
        self.ball_dist_sum = self._calc_ball_dist_sum()

    def on_step_end(self):
        super(PushNegBallsFromGoal, self).on_step_end()
        ball_dist_delta = self._calc_ball_dist_sum() - self.ball_dist_sum
        self.reward_happy -= ball_dist_delta * const.POINTS_BALL_TRAVEL_MULT
        self.reward_grumpy += ball_dist_delta * const.POINTS_BALL_TRAVEL_MULT

    def _calc_ball_dist_sum(self):
        # Grumpy's goal is in the 0,0 corner, therefore higher distance is better for Grumpy team
        return sum(map(lambda x: distance((0, 0), x.rectDbl.center), self.lstPosBalls))
