import pygame

ARENA_WIDTH = 800
ARENA_HEIGHT = 800
ROBOT_LENGTH = 20
ROBOT_WIDTH= 40
ROBOT_WIDTH_BODY = 32
ROBOT_WIDTH_TRACKS = 8
ROBOT_VEL = 3
ROBOT_ANGULAR_VEL_ONE = .5  #.6 # one motor rotating around other track
ROBOT_ANGULAR_VEL_BOTH = 1.5  # with both motors engaged
GOAL_WIDTH = 160
GOAL_HEIGHT = 160

BALL_RADIUS = 7
BALL_SLOWDOWN = .995
FRAMERATE = 120
GAME_LENGTH_MINS = 3
GAME_LENGTH_STEPS = GAME_LENGTH_MINS * 60 * FRAMERATE

NUM_BALL_POS = 4
NUM_BALL_NEG = 4
NUM_ROBOTS_HAPPY = 1
NUM_ROBOTS_GRUMPY = 1

TEAM_HAPPY = 1
TEAM_GRUMPY = -1

# NN controls (most likely)
KEY_LEFT_MOTOR_FORWARD = pygame.K_w
KEY_LEFT_MOTOR_BACKWARD = pygame.K_s
KEY_RIGHT_MOTOR_FORWARD = pygame.K_o
KEY_RIGHT_MOTOR_BACKWARD = pygame.K_l

# Sane human controls
KEY_BOTH_MOTOR_FORWARD = pygame.K_UP
KEY_BOTH_MOTOR_BACKWARD = pygame.K_DOWN
KEY_BOTH_MOTOR_LEFT = pygame.K_LEFT
KEY_BOTH_MOTOR_RIGHT = pygame.K_RIGHT

# Colors
COLOR_BACKGROUND = (255,255,255)
COLOR_ROBOT_BODY = (255,0,0)
COLOR_ROBOT_TRACK = (128,128,128)
COLOR_BALL_POS = (80,220,100)
COLOR_BALL_NEG = (60,16,83)
COLOR_GOAL_HAPPY = (43, 146, 228)
COLOR_GOAL_GRUMPY = (242, 53, 87)

# Calculation parameters
CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER = (ROBOT_WIDTH/2) - (ROBOT_WIDTH_TRACKS/2)

BOUNCE_K_WALL = 1 # just adding it in case later we want it

# # distance from center track to center robot, cubed, and flipped
# # used often to rotate around a track when one motor engaged
# # do not need to recalc each time (hence adding as shared const)
# CALC_ROTATION_TERM_1 = -1 * math.pow(CALC_DIST_TRACK_CENTER_TO_ROBOT_CENTER, 3)
