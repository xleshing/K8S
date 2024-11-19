import random

# snake size
SIZE = 20

# ranges for defining close and far
CLOSE_RANGE = (0, 15)
FAR_RANGE = (CLOSE_RANGE[1], 30)


DEFAULT_WINDOW_SIZES = (32, 24)

DEFAULT_KILL_FRAME = 200
DEFAULT_SPEED = 50  # change the speed of the game
DEFAULT_N_FOOD = 1
DECREASE_FOOD_CHANCE = 0.8
BAD_REWARD = 0.98
GOOD_REWARD = 1.02

# Neural Networks Configuration
OUTPUT_SIZE = 3

MAX_MEMORY = 100_000
BATCH_SIZE = 2000
LR = 0.001

GAMMA = 0.9

EPSILON = 80

EPS_RANGE = (0, 300)

is_random_move = lambda eps, eps_range: random.randint(eps_range[0], eps_range[1]) < eps

DEFAULT_END_GAME_POINT = 300
