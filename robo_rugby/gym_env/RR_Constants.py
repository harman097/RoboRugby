import pygame
import math

ARENA_WIDTH = 800
ARENA_HEIGHT = 800
ROBOT_LENGTH = 20
ROBOT_WIDTH= 40
ROBOT_WIDTH_BODY = 32
ROBOT_WIDTH_TRACKS = 8
ROBOT_VEL = 3 # todo can probly delete this and move_all()
MOVES_PER_FRAME = ROBOT_VEL
ROBOT_ANGULAR_VEL_ONE = .9  #.6 # one motor rotating around other track
ROBOT_ANGULAR_VEL_BOTH = 1.5  # with both motors engaged
GOAL_WIDTH = 240
GOAL_HEIGHT = 240

BALL_RADIUS = 7
BALL_SLOWDOWN = .995
BALL_MIN_SPEED = 0.005
PUSH_FACTOR = BALL_SLOWDOWN
FRAMERATE = 120
GAME_LENGTH_MINS = 3
GAME_LENGTH_STEPS = GAME_LENGTH_MINS * 60 * FRAMERATE
TIME_BALL_IN_GOAL_SECONDS = 5
TIME_BALL_IN_GOAL_STEPS = TIME_BALL_IN_GOAL_SECONDS * FRAMERATE

NUM_BALL_POS = 4
NUM_BALL_NEG = 4
NUM_ROBOTS_HAPPY = 2
NUM_ROBOTS_GRUMPY = 2
NUM_ROBOTS_TOTAL = NUM_ROBOTS_GRUMPY + NUM_ROBOTS_HAPPY

MAX_NEG_BALLS = 3
POINTS_BALL_SCORED = 1000
POINTS_TIME_PENALTY = .1 # Time penalty per frame
POINTS_ROBOT_CRASH_PENALTY = 10
POINTS_ROBOT_IN_GOAL_PENALTY = 10
# Let's say that if a ball starts at (0,0) and travels all the way to (ARENA_WIDTH,ARENA_HEIGHT)
# the ball would get this many points:
POINTS_BALL_TRAVEL_MAX = 1000
# Then for each pixel moved closer to the goal it gets:
POINTS_BALL_TRAVEL_MULT = POINTS_BALL_TRAVEL_MAX / math.pow(ARENA_WIDTH**2 + ARENA_HEIGHT**2, .5)
# If you destroy a goal, get all the points you possibly could from dunking all the balls
POINTS_GOAL_DESTROYED = (POINTS_BALL_SCORED + POINTS_BALL_TRAVEL_MAX) * (NUM_BALL_POS + NUM_BALL_NEG)

TEAM_HAPPY = 1
TEAM_GRUMPY = -1

# NN controls (most likely)
KEY_LEFT_MOTOR_FORWARD = pygame.K_i
KEY_LEFT_MOTOR_BACKWARD = pygame.K_k
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

BOUNCE_K_WALL = .8 # just adding it in case later we want it
BOUNCE_K_ROBOT = .5
BOUNCE_K_BALL = .995

# "Mass" (so to speak)
# In the world of trashy physics, you just straight up can't
# affect the momentum of something with higher mass
MASS_BALL = 1
MASS_ROBOT = 2
MASS_WALL = 3
