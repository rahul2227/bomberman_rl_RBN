# This file will contain the code for the helper functions and definitions
# needed in the code calculations or global definitions
import itertools

import numpy as np

# Actions
ACTIONS_ALL = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
BOMB_MOVES = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1),
    'SAFE': (0, 0)
}

# MAKING GENERALIZED EVENTS FOR DIRECTIONS
UP = 'UP'
DOWN = 'DOWN'
LEFT = 'LEFT'
RIGHT = 'RIGHT'

# HIT DIRECTION STATE FOR OBSTACLE
NO_OBSTACLE = 'NO_OBSTACLE'

# COIN EVENT
COIN_COLLECTED = 'COIN_COLLECTED'
MOVED_TO_COIN = 'MOVED_TO_COIN'
MOVED_AWAY_COIN = 'MOVED_AWAY_COIN'
COIN_MISSED = 'COIN_MISSED'

# EVENT FOR OBSTACLE HIT OR NOT
OBSTACLE_HIT = 'OBSTACLE_HIT'
OBSTACLE_AVOID = 'OBSTACLE_AVOID'

# Good Move or Bad Move
GOOD_MOVE = 'GOOD_MOVE'
BAD_MOVE = 'BAD_MOVE'

# BOMB_ESCAPE
ESCAPE_BOMB_YES = 'ESCAPE_BOMB_YES'
ESCAPE_BOMB_NO = 'ESCAPE_BOMB_NO'


# This is a state index
action_index = {
    UP: 0,
    DOWN: 1,
    LEFT: 2,
    RIGHT: 3,
    NO_OBSTACLE: 4,
    COIN_COLLECTED: 5,
    COIN_MISSED: 7
}


def valid_action() -> np.array:
    features = []

    valid_states = list(itertools.product(
        ['MOVE_UP', 'OBSTACLE'],
        ['MOVE_DOWN', 'OBSTACLE'],
        ['MOVE_LEFT', 'OBSTACLE'],
        ['MOVE_RIGHT', 'OBSTACLE'],
        # ['COIN_UP', 'COIN_DOWN', 'COIN_LEFT', 'COIN_RIGHT', 'NO_COIN']  # This is the state to coin direction
        ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NO_COIN'],  # These are the action I should take upon coin encounter
        ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SAFE'],
    ))

    # This will create every possible combination of the states for the features
    for state in valid_states:
        feature = {
            'UP': state[0],
            'DOWN': state[1],
            'LEFT': state[2],
            'RIGHT': state[3],
            'COIN_DIRECTION': state[4],
            'BOMB_ESCAPE': state[5],
        }
        features.append(feature)

    return features
