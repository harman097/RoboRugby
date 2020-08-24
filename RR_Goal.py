import pygame
import RR_Constants as const

class Goal(pygame.sprite.Sprite):
    def __init__(self, intTeam):
        super(Goal, self).__init__()
        self.intTeam = intTeam
        self.surf = pygame.Surface((const.GOAL_WIDTH, const.GOAL_HEIGHT))
        if intTeam == const.TEAM_HAPPY:
            self.tplColor = const.COLOR_GOAL_HAPPY
            # Bottom right
            self.tplCenter = (const.ARENA_WIDTH - const.GOAL_WIDTH/2,
                              const.ARENA_HEIGHT - const.GOAL_HEIGHT/2)
        else:
            self.tplColor = const.COLOR_GOAL_GRUMPY
            self.tplCenter = (const.GOAL_WIDTH/2, const.GOAL_HEIGHT/2)

        self.surf.fill(self.tplColor)
        self.rect = self.surf.get_rect(center=self.tplCenter)
