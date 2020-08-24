from __future__ import absolute_import, division, print_function

import MyUtils
from MyUtils import Stage
import gym
import pygame
import RR_Constants as const

Stage("Initialize pygame")
pygame.init()
mScreen = pygame.display.set_mode((const.ARENA_WIDTH, const.ARENA_HEIGHT))

from RR_Ball import Ball
from RR_Robot import Robot
from RR_Goal import Goal
import RR_TrashyPhysics as TrashyPhysics

MyUtils.PRINT_STAGE = False  # Disable stage spam

# TODO perf stuff (if we end up doing serious training with this)
# (1) Pre-cache sin/cos results in a dictionary when initializing
# Avoid math.sin/cos and radian->degree conversion (cuz it's probly slow?)

# TODO properly inherit
class GameEnv(gym.Env):
    def __init__(self):
        Stage("Initializing RoboRugby...")
        self.grpAllSprites = pygame.sprite.Group()
        self.lngStepCount = 0

        # TODO set these properly (if you want to properly implement gym.Env)
        # self.action_space = gym.spaces.Tuple()? .MultiDiscrete()
        # self.observation_space = gym.spaces.MultiDiscrete()? Other?

        Stage("Here are the goals")
        self.sprHappyGoal = Goal(const.TEAM_HAPPY)
        self.sprGrumpyGoal = Goal(const.TEAM_GRUMPY)
        self.grpAllSprites.add(self.sprHappyGoal)
        self.grpAllSprites.add(self.sprGrumpyGoal)

        Stage("Bring out ze Robots!")
        self.grpRobots = pygame.sprite.Group()
        lngTotalRobots = const.NUM_ROBOTS_GRUMPY + const.NUM_ROBOTS_HAPPY
        self.lstRobots = [0] * (lngTotalRobots)
        for i in range(const.NUM_ROBOTS_HAPPY):
            self.lstRobots[i] = Robot(const.TEAM_HAPPY)
            while pygame.sprite.spritecollideany(self.lstRobots[i], self.grpAllSprites):
                # Keep trying random start positions until there's no overlap
                self.lstRobots[i] = Robot(const.TEAM_HAPPY)
            self.grpRobots.add(self.lstRobots[i])
            self.grpAllSprites.add(self.lstRobots[i])

        for i in range(const.NUM_ROBOTS_HAPPY, lngTotalRobots):
            self.lstRobots[i] = Robot(const.TEAM_GRUMPY)
            while pygame.sprite.spritecollideany(self.lstRobots[i], self.grpAllSprites):
                # Keep trying random start positions until there's no overlap
                self.lstRobots[i] = Robot(const.TEAM_GRUMPY)
            self.grpRobots.add(self.lstRobots[i])
            self.grpAllSprites.add(self.lstRobots[i])

        Stage("Give it some balls!")
        self.grpBalls = pygame.sprite.Group()
        for _ in range(const.NUM_BALL_POS):
            while True:  # Just make one until we have no initial collisions
                objNewBall = Ball(const.COLOR_BALL_POS)
                if pygame.sprite.spritecollideany(objNewBall, self.grpAllSprites) is None:
                    break
            self.grpBalls.add(objNewBall)
            self.grpAllSprites.add(objNewBall)

        for _ in range(const.NUM_BALL_NEG):
            while True:  # Just make one until we have no initial collisions
                objNewBall = Ball(const.COLOR_BALL_NEG)
                if pygame.sprite.spritecollideany(objNewBall, self.grpAllSprites) is None:
                    break
            self.grpBalls.add(objNewBall)
            self.grpAllSprites.add(objNewBall)

    def reset(self):
        raise NotImplementedError("Probly just move most of the logic from __init__ to here")
        return self._get_game_state()

    def render(self):
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

    def step(self, *lstArgs):
        if self.game_is_done():
            raise Exception("Game is over. Go home.")
        else:
            self.lngStepCount += 1

        Stage("Prep ze robots!")
        for sprRobot in self.lstRobots:
            sprRobot.on_step_begin()

        Stage("Activate robot engines!")
        for i in range(len(lstArgs)):
            self.lstRobots[i].set_thrust(lstArgs[i][0], lstArgs[i][1])

        Stage("Move those robots!")
        for sprRobot in self.grpRobots.sprites():
            sprRobot.move_all()

        Stage("Did they smash each other?")
        lngNaughtyLoop = 0
        while True:
            lngNaughtyLoop += 1
            if lngNaughtyLoop > 5:
                raise Exception("Screw this.")
            blnNaughtyBots = False
            
            for sprRobot1 in self.grpRobots.sprites():
                for sprRobot2 in pygame.sprite.spritecollide(sprRobot1, self.grpRobots, False, collided=TrashyPhysics.check_robot_robot_collision):
                    # TODO deduct some points here but... who gets deducted? "who smashed who" should matter...
                    if not sprRobot1 is sprRobot2:
                        blnNaughtyBots = True
                        TrashyPhysics.resolve_robot_robot_collision(sprRobot1, sprRobot2)
                        
            if not blnNaughtyBots:
                break                

        Stage("Let the balls roll around")
        for sprBall in self.grpBalls.sprites():
            sprBall.move()

        Stage("Push those balls! like with a robot")
        for sprBall in self.grpBalls.sprites():
            lstCollisions = pygame.sprite.spritecollide(sprBall, self.grpRobots, False, collided=TrashyPhysics.check_ball_robot_collision)
            if len(lstCollisions) > 1 or \
                    (len(lstCollisions) == 1 and TrashyPhysics.check_collision_wall(sprBall)):
                # robots are smashing into same ball (without smashing each other) -or-
                # robot is smashing ball into wall
                raise NotImplementedError("Back the robots up, man!")

            elif len(lstCollisions) == 1:  # just push the ball
                objRobot = lstCollisions[0]

                # TODO pick this up here

                # Determine the vector the robot has actually traveled (don't trust thrust)

                # Trashy trash solely so it's interesting, for the moment
                lngDeltaX = objRobot.rect.centerx - objRobot.rectPrior.centerx
                lngDeltaY = objRobot.rect.centery - objRobot.rectPrior.centery

                if lngDeltaX > 0:
                    sprBall.dblXVelocity = max(lngDeltaX, sprBall.dblXVelocity)
                else:
                    sprBall.dblXVelocity = min(lngDeltaX, sprBall.dblXVelocity)

                if lngDeltaY > 0:
                    sprBall.dblYVelocity = max(lngDeltaY, sprBall.dblYVelocity)
                else:
                    sprBall.dblYVelocity = min(lngDeltaY, sprBall.dblYVelocity)

                # TODO this is absolute trash
                sprBall.rect.move_ip(lngDeltaX, lngDeltaY)

        # Pseudo "good enough" code
        # Check for robot-on-ball collision
        #   Resolve by:
        #       flag balls being pushed by robots
        #       assigning velocity based on the robot's velocity (not thrust)
        #       popping the ball out of the robot (linear should be easy)
        #       if the ball gets pushed into a wall... just stop the Robot and "disallow" this
        #           pop the ball out of the wall
        #           back up the robot along it's vector
        #       if the ball gets pushed into another ball...
        #           calc vector from center pusher to center pushee
        #           don't move the pusher (if it's attached to a robot - unrealistic, but good enough?)
        #           move the pushee along the vector until it's no longer overlapped
        # Move balls not being pushed
        #
        # Probly stuff this whole thing in a loop with a very low loop limit


        return self._get_game_state(), self._get_reward(), self.game_is_done(), self._get_debug_info()

    def _get_reward(self):
        # TODO this
        return 1

    def _get_game_state(self):
        # TODO this
        return 1

    def game_is_done(self):
        return self.lngStepCount > const.GAME_LENGTH_STEPS

    def _get_debug_info(self):
        # currently no need for this, but something will probly come up
        return None





