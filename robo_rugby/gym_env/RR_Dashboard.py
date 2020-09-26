import pygame
from . import RR_Constants as const
from typing import List
from robo_rugby.gym_env.RR_Ball import Ball
from MyUtils import RightTriangle, Point

class Dashboard(pygame.sprite.Sprite):
    def __init__(self):
        super(Dashboard, self).__init__()        
        pygame.font.init()
        self.font_title = pygame.font.SysFont(const.DASHBOARD_FONT, const.DASHBOARD_FONT_SIZE_TITLE)
        self.font_info = pygame.font.SysFont(const.DASHBOARD_FONT, const.DASHBOARD_FONT_SIZE_INFO)
        
    @property
    def surf(self) -> pygame.Surface:
        return pygame.Surface((const.DASHBOARD_WIDTH, const.ARENA_HEIGHT))      

    @property
    def border_rect(self) -> pygame.Rect:
        return self.surf.get_rect(left=const.ARENA_WIDTH, top=0)       


    @property
    def rect(self) -> pygame.Rect:
        return self.border_rect.inflate(-const.DASHBOARD_BORDER_WIDTH, -const.DASHBOARD_BORDER_WIDTH)


    def update(self, screen: pygame.surface, dic_info: dict):
        #background and boarder of dashboard
        pygame.draw.rect(screen,const.COLOR_DASHBOARD_BORDER,self.border_rect)
        pygame.draw.rect(screen,const.COLOR_DASHBOARD_FILL,self.rect)
     
        #title and info to display
        self.set_title(screen, 'DASHBOARD')
        self.set_info(screen, dic_info)

        #TODO: Could possibly draw line path calculations, like lidar, or path finding if we implement that
        #TODO: 

        
    def set_title(self, screen: pygame.surface, text: str):
        text_surf = self.font_title.render(text, True, const.COLOR_DASHBOARD_TEXT)
        
        text_rect = text_surf.get_rect()
        text_rect.top = const.DASHBOARD_BORDER_WIDTH + 20
        text_rect.left = (const.ARENA_WIDTH + 
                         const.DASHBOARD_BORDER_WIDTH + 
                         (self.rect.width/2 - text_rect.width/2))
        screen.blit(text_surf, text_rect)


    def set_info(self, screen: pygame.surface, dic_info: dict):               
        width_offset = 20
        height_offset = 60
        index_offset = 20

        max_key_len = max(map(len, dic_info))
        max_value_len = max(dic_info.values(), key=lambda x:len(str(x)))
        for index, (key, value)  in enumerate(dic_info.items()):
            text = str(key).ljust(max_key_len + 1) + ': ' + str(value).rjust(max_value_len + 1) 
            
            text_surf = self.font_info.render(text, 
                                              True, 
                                              const.COLOR_DASHBOARD_TEXT)
            text_rect = text_surf.get_rect()
            text_rect.top = const.DASHBOARD_BORDER_WIDTH + height_offset + index * index_offset
            text_rect.left = (const.ARENA_WIDTH + 
                              const.DASHBOARD_BORDER_WIDTH + 
                              width_offset)
            if text_rect.right > self.border_rect.right:
                raise Exception('\n  Real ugly. Smaller font, or shorter name, or something.' +
                                '\n  Will somebody please think of the children?!?' +
                                '\n  ' + text)
            screen.blit(text_surf, text_rect)
                
        



    
        
        
    