import pygame
import math

GAME_MODE = False

ARENA_WIDTH = 800 if GAME_MODE else 800
ARENA_HEIGHT = 800 if GAME_MODE else 800
ROBOT_LENGTH = 20
ROBOT_WIDTH = 40
ROBOT_WIDTH_BODY = 32
ROBOT_WIDTH_TRACKS = 8
ROBOT_VEL = 12 if GAME_MODE else 12  # Std game mode = 3
MOVES_PER_FRAME = ROBOT_VEL
ROBOT_ANGULAR_VEL_ONE = 3.6  # Std game mode .9  # one motor rotating around other track
ROBOT_ANGULAR_VEL_BOTH = 6  # Std game mode 1.5  # with both motors engaged
GOAL_WIDTH = 240
GOAL_HEIGHT = 240

BALL_RADIUS = 7
BALL_SLOWDOWN = .995
BALL_MIN_SPEED = 0.005
PUSH_FACTOR = BALL_SLOWDOWN
FRAMERATE = 30 if GAME_MODE else 30  # Std game mode = 120
GAME_LENGTH_MINS = 3 if GAME_MODE else .4 # Std training so far = .15
GAME_LENGTH_STEPS = int(GAME_LENGTH_MINS * 60 * FRAMERATE)

TIME_BALL_IN_GOAL_SECONDS = 5
TIME_BALL_IN_GOAL_STEPS = TIME_BALL_IN_GOAL_SECONDS * FRAMERATE

NUM_BALL_POS = 4 if GAME_MODE else 1
NUM_BALL_NEG = 4 if GAME_MODE else 0
NUM_ROBOTS_HAPPY = 2 if GAME_MODE else 3
NUM_ROBOTS_GRUMPY = 2 if GAME_MODE else 3
NUM_ROBOTS_TOTAL = NUM_ROBOTS_GRUMPY + NUM_ROBOTS_HAPPY

MAX_NEG_BALLS = 3
POINTS_BALL_SCORED = 500
POINTS_TIME_PENALTY = .1  # Time penalty per frame
POINTS_ROBOT_CRASH_PENALTY = .005
POINTS_ROBOT_IN_GOAL_PENALTY = .005
POINTS_NO_MOVE_PENALTY = .005
# Let's say that if a ball starts at (0,0) and travels all the way to (ARENA_WIDTH,ARENA_HEIGHT)
# the ball would get this many points:
POINTS_BALL_TRAVEL_MAX = 200000
# Then for each pixel moved closer to the goal it gets:
POINTS_BALL_TRAVEL_MULT = POINTS_BALL_TRAVEL_MAX / math.pow(ARENA_WIDTH**2 + ARENA_HEIGHT**2, .5)
# If you destroy a goal, get all the points you possibly could from dunking all the balls
POINTS_GOAL_DESTROYED = (POINTS_BALL_SCORED + POINTS_BALL_TRAVEL_MAX) * (NUM_BALL_POS + NUM_BALL_NEG)

POINTS_ROBOT_TRAVEL_MULT = POINTS_BALL_TRAVEL_MULT / 100  # for now

TEAM_HAPPY = 1
TEAM_GRUMPY = -1

# NN controls (most likely)
KEY_LEFT_MOTOR_FORWARD = pygame.K_i
KEY_LEFT_MOTOR_BACKWARD = pygame.K_k
KEY_RIGHT_MOTOR_FORWARD = pygame.K_o
KEY_RIGHT_MOTOR_BACKWARD = pygame.K_l

# Sane human controls
KEY_BOTH_MOTOR_FORWARD = pygame.K_w
KEY_BOTH_MOTOR_BACKWARD = pygame.K_s
KEY_BOTH_MOTOR_LEFT = pygame.K_a
KEY_BOTH_MOTOR_RIGHT = pygame.K_d

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
