from __future__ import absolute_import, division, print_function

from MyUtils import Stage
import pygame
import random
import imageio
import RoboRugby
from RoboRugby import const

# Each frame of the game is driven by step()
#
# So we need to design our game in a way that it can handle
# keyboard input vs machine input to the step() function.

blnPLAYER = True  # person is playing
blnRENDER = True  # should we slow it down and render?
blnRECORD = True
strVideoFile = "SeeHowAwesomeYouDid.mp4"
if blnPLAYER:
    blnRENDER = True

if blnRENDER:
    objClock = pygame.time.Clock()

objVideoWriter = None
if blnRECORD:
    objVideoWriter = imageio.get_writer(strVideoFile, fps=RoboRugby.GameEnv.metadata['video.frames_per_second'])

Stage("Start the game!")
env = RoboRugby.GameEnv(RoboRugby.GameEnv.CONFIG_STANDARD)
blnRunGame = True
while blnRunGame:

    if blnPLAYER:
        # Check for when player is quitting
        for objEvent in pygame.event.get():
            if objEvent.type == pygame.QUIT:
                blnRunGame = False
                break
            elif objEvent.type == pygame.KEYDOWN and objEvent.key == pygame.K_ESCAPE:
                blnRunGame = False
                break

        # Process player input to send to the step function
        dctKeyDown = pygame.key.get_pressed()
        lngLThrust = 0
        lngRThrust = 0
        if dctKeyDown[const.KEY_LEFT_MOTOR_FORWARD]: lngLThrust += 1
        if dctKeyDown[const.KEY_LEFT_MOTOR_BACKWARD]: lngLThrust -= 1
        if dctKeyDown[const.KEY_RIGHT_MOTOR_FORWARD]: lngRThrust += 1
        if dctKeyDown[const.KEY_RIGHT_MOTOR_BACKWARD]: lngRThrust -= 1
        if dctKeyDown[const.KEY_BOTH_MOTOR_FORWARD]:
            lngLThrust += 1
            lngRThrust += 1
        if dctKeyDown[const.KEY_BOTH_MOTOR_BACKWARD]:
            lngLThrust -= 1
            lngRThrust -= 1
        if dctKeyDown[const.KEY_BOTH_MOTOR_LEFT]:
            lngLThrust -= 1
            lngRThrust += 1
        if dctKeyDown[const.KEY_BOTH_MOTOR_RIGHT]:
            lngLThrust += 1
            lngRThrust -= 1

        lstInput = [(lngLThrust, lngRThrust)]
        for _ in range(1, const.NUM_ROBOTS_HAPPY + const.NUM_ROBOTS_GRUMPY):
            rando = random.random()
            action = (0,0)
            # ~5% chance to turn left or right, 45% chance to go forward/back
            if rando <= 0.05:
                # turn left
                action = (-1,1)
            elif rando <= 0.5:
                # go straight
                action = (1,1)
            elif rando < 0.95:
                # go back
                action = (-1,-1)
            else:
                # turn right
                action = (1,-1)

            # (random.randint(-1,1), random.randint(-1,1))
            lstInput.append(action)

        observation, reward, done, info  = env.step(lstInput)

        if done:
            # blnRunGame = False

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

            print("------ RESETTING --------")
            env.reset(False)

    else:
        raise NotImplementedError("Training section not done.")

    if blnRENDER:
        if blnRECORD:
            objVideoWriter.append_data(env.render(mode='rgb_array'))
        else:
            env.render(mode='human')
        # Ensure we maintain a framerate of 120 fps
        objClock.tick(const.FRAMERATE)

if objVideoWriter is not None:
    objVideoWriter.close()
Stage("Bye!")
pygame.quit()
