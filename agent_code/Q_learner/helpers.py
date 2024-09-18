# This file will contain the code for the helper functions and definitions
# needed in the code calculations or global definitions


# Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Events for Wall encounter
WALL_AVOIDED = 'WALL_AVOIDED'
WALL_HIT = 'WALL_HIT'
# Events for Wall presence
WALL_UP = 'WALL_UP'
WALL_DOWN = 'WALL_DOWN'
WALL_LEFT = 'WALL_LEFT'
WALL_RIGHT = 'WALL_RIGHT'
NO_OBSTACLE = 'NO_OBSTACLE'
OBSTACLE_HIT = 'OBSTACLE_HIT'
OBSTACLE_AVOID = 'OBSTACLE_AVOID'

wall_action_index = {
    WALL_UP: 0,
    WALL_DOWN: 1,
    WALL_LEFT: 2,
    WALL_RIGHT: 3,
    NO_OBSTACLE: 4,
}