from __future__ import absolute_import, division, print_function

from MyUtils import Stage
import pygame
import random
import RoboRugby
from RoboRugby import const

# Each frame of the game is driven by step()
#
# So we need to design our game in a way that it can handle
# keyboard input vs machine input to the step() function.

blnPLAYER = True  # person is playing
blnRENDER = True  # should we slow it down and render?
if blnPLAYER:
    blnRENDER = True

if blnRENDER:
    objClock = pygame.time.Clock()

Stage("Start the game!")
env = RoboRugby.GameEnv()
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

        env.step((lngLThrust, lngRThrust))
    else:
        raise NotImplementedError("Training section not done.")

    if blnRENDER:
        env.render()
        # Ensure we maintain a framerate of 120 fps
        objClock.tick(const.FRAMERATE)

Stage("Bye!")
pygame.quit()
