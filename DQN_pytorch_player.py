from robo_rugby.gym_env.RR_Players import AbstractPlayer
from robo_rugby.gym_env import GameEnv, Robot, Ball, RR_Observers, GameEnv_Simple
import pickle
import Training_DQN_pytorch
from Training_DQN_pytorch import DQNAgent, DeepQNetwork
from typing import Tuple, List, Dict, Set
from MyUtils import distance


class Stephen(AbstractPlayer):
    __hive = set()  # type: Set['Stephen']
    __mind = None  # type: DQNAgent
    __last_assessed = -1
    __assignments = {}
    __env = None  # type: GameEnv

    @staticmethod
    def __embrace(brother: 'Stephen'):
        Stephen.__hive.add(brother)

        if Stephen.__mind is None:
            Stephen.__awaken()

        if Stephen.__env is None:
            Stephen.__env = brother.env
        elif Stephen.__env is not brother.env:
            raise Exception("He is not one of us.")

    @staticmethod
    def __awaken():
        str_env = "RoboRugbySimpleDuel-v3"
        lng_start_episode = 6250
        str_session = "2020_09_21__09_36"
        dir = f"DQN_Pytorch_{str_env}_{str_session}"
        with open(f"checkpoints/dqn/{dir}/{dir}_Ep_{lng_start_episode}.pickle", "rb") as f:
            Stephen.__mind = pickle.load(file=f)

    @staticmethod
    def __ponder():
        Stephen.__assignments = {}
        lst_dist = []
        for ball in Stephen.__env.lstBalls:
            if Stephen.__env.sprGrumpyGoal.ball_in_goal(ball) or \
                    Stephen.__env.sprHappyGoal.ball_in_goal(ball):
                pass  # ignore
            else:
                for stephen in Stephen.__hive:
                    lst_dist.append(
                        (stephen, ball, distance(ball.rectDbl.center, stephen.robot.rectDbl.center))
                    )

        lst_dist.sort(key=lambda s: s[2])  # sort by distance, asc
        claimed_balls = set()
        for stephen, ball, _ in lst_dist:
            if ball in claimed_balls:
                pass
            elif stephen in Stephen.__assignments:
                pass
            else:
                claimed_balls.add(ball)
                Stephen.__assignments[stephen] = ball

    @staticmethod
    def __consult(brother: 'Stephen') -> Tuple[float, float]:
        if Stephen.__last_assessed != Stephen.__env.lngStepCount:
            Stephen.__ponder()

        if not brother in Stephen.__assignments:
            return (0, 0)

        obs = Stephen.__env.get_game_state(obj_robot=brother.robot, obj_ball=Stephen.__assignments[brother])
        action = Stephen.__mind.choose_action(obs, epsilon_override=0.2)
        return GameEnv_Simple.thrust_from_direction(action)

    def __init__(self, env: GameEnv, robot: Robot):
        super(Stephen, self).__init__(env, robot)
        if not isinstance(env, RR_Observers.SingleBall_6wayLidar):
            raise Exception("""
            Stephen can only be added to environments that support 'RR_Observers.SingleBall_6wayLidar' observations.
            Create an instance of the game that inherits from 'RR_Observers.SingleBall_6wayLidar'.
            """)
        Stephen.__embrace(self)

    def get_action(self) -> Tuple[float, float]:
        return Stephen.__consult(self)  # consult the Multi-Stephen
