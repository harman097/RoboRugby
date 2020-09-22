from __future__ import absolute_import, division, print_function

from MyUtils import Stage
import pygame
import random
import imageio
import robo_rugby

""" Needed in order to de-pickle the model saved from Training_DQN_pytorch (pickle quirk)"""
import pickle
import Training_DQN_pytorch
from Training_DQN_pytorch import DQNAgent, DeepQNetwork

from robo_rugby.gym_env.RR_EnvBase import GameEnv
import robo_rugby.gym_env.RR_Observers as RR_Observers
import robo_rugby.gym_env.RR_ScoreKeepers as RR_Scorekeepers
from robo_rugby.gym_env.RR_Players import Human, OG_Twitchy
from DQN_pytorch_player import Stephen
import robo_rugby.gym_env.RR_Constants as const

# Each frame of the game is driven by step()
#
# So we need to design our game in a way that it can handle
# keyboard input vs machine input to the step() function.

blnRECORD = False
strVideoFile = "SeeHowAwesomeYouDid.mp4"

objClock = pygame.time.Clock()

objVideoWriter = None
if blnRECORD:
    objVideoWriter = imageio.get_writer(strVideoFile, fps=GameEnv.metadata['video.frames_per_second'])

Stage("Start the game!")
if not const.GAME_MODE:
    """ If anyone has a better way of managing this, I'm all ears. Doing this for now since it's easiest. """
    print("WARNING: RoboRugby constants aren't set for game mode. Turn on the GAME_MODE flag in RR_Constants.")
# env = GameEnv()

""" Create a custom instance with the components we want """
class JustPlayTheFreakingGame(
    RR_Scorekeepers.PushPosBallsInYourGoal,
    RR_Scorekeepers.PushNegBallsInTheirGoal,
    RR_Scorekeepers.BaseDestruction,
    RR_Observers.SingleBall_6wayLidar,  # so Stephen can play
    GameEnv  # not stricly necessary, since all these other inherit from it, as well
):
    pass

env = JustPlayTheFreakingGame()
blnRunGame = True
total_reward = 0

"""
INTRODUCE THE PLAYERS!!!!
"""
lstPlayers = [
    Stephen(env, env.lstHappyBots[0]),
    Stephen(env, env.lstHappyBots[1]),
    Human(env, env.lstGrumpyBots[0], key_left=pygame.K_a, key_right=pygame.K_d,
          key_forwards=pygame.K_w, key_backwards=pygame.K_s),
    OG_Twitchy(env, env.lstGrumpyBots[1])
]

while blnRunGame:

    # Check for when player is quitting
    for objEvent in pygame.event.get():
        if objEvent.type == pygame.QUIT:
            blnRunGame = False
            break
        elif objEvent.type == pygame.KEYDOWN and objEvent.key == pygame.K_ESCAPE:
            blnRunGame = False
            break

    lstActions = list(map(lambda x: x.get_action(), lstPlayers))

    observation, reward, done, info = env.step(lstActions)
    total_reward += reward

    if blnRECORD:
        objVideoWriter.append_data(env.render(mode='rgb_array'))
    else:
        env.render(mode='human')

    # Ensure we maintain a framerate of ... whatever the framerate is (when rendering)
    objClock.tick(const.FRAMERATE)

    if done:
        # blnRunGame = False  NOPE, PLAY MOAR!
        lngScoreHappy = env.sprHappyGoal.get_score()
        lngScoreGrumpy = env.sprGrumpyGoal.get_score()
        print("\nFinal Score")
        print("Happy Bot %1d  |  Grumpy Bot %1d" % (lngScoreHappy, lngScoreGrumpy))
        if lngScoreHappy > lngScoreGrumpy:
            print("Team Happy Bot WINS!!!")
        elif lngScoreHappy == lngScoreGrumpy:
            print("Tie!")
        else:
            print("Team Grump Bot WINS!!!  ~~boo~~")

        print(f"Reward: {total_reward}")

        print("------ RESETTING --------")
        total_reward = 0
        env.reset(False)

if objVideoWriter is not None:
    objVideoWriter.close()
Stage("Bye!")
pygame.quit()
