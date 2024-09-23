import os
import random
from collections import deque

import numpy as np

from agent_code.Q_learner.helpers import ACTIONS, valid_action, BOMB_MOVES, calculate_decay_rate


def setup(self):
    q_table_folder = "q_tables/"
    self.new_state = None
    self.old_state = None
    self.valid_actions_list = valid_action(self)
    if self.train:
        self.logger.info("Q-learning agent from scratch")
        self.number_of_actions = len(self.valid_actions_list)
        self.Q_table = np.zeros(shape=(self.number_of_actions, len(ACTIONS)))
        self.logger.debug(f"Q-table is {self.Q_table.shape}")
        # self.logger.info(f"Initialized Q-table = {self.Q_table}")
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
        self.exploration_rate_initial = 1.0  # random choice
        self.exploration_rate_end = 0.02
        self.exploration_rate = calculate_decay_rate(self)
    else:
        self.logger.info("Loading q-table from saved state.")
        self.Q_table = load_q_table(self, q_table_folder)


def act(self, game_state: dict) -> str:
    # updating the custom state so that features can be accessible
    if self.new_state is None:
        self.old_state = state_to_features(self, game_state)
    else:
        self.old_state = self.new_state

    # Exploration vs exploitation
    if self.train and self.exploration_rate > random.random():
        self.logger.debug("Choosing action purely at random.")
        # taking a random action for exploration
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    # Querying the Q-table for an action
    # action = np.random.choice(ACTIONS)
    action = ACTIONS[np.argmax(self.Q_table[self.old_state])]
    self.logger.debug(f"action taken by the q-table model: {action}")
    return action


def state_to_features(self, game_state: dict) -> np.array:
    features = {}

    # FEATURE1: checking for obstacles
    # this returns a list of walls in the agent's vicinity
    obstacles = check_the_presence_of_walls(self, game_state)
    for direction in obstacles:
        if direction in ACTIONS:
            features[direction] = 'OBSTACLE'

    # for the remaining direction
    directions = find_missing_directions(self, obstacles)
    for direction in directions:
        features[direction] = f'MOVE_{direction}'

    # FEATURE2: checking for coin presence and chasing coin
    coins_present, nearest_coin = check_for_coin_presence(self, game_state)

    nearest_coin_path = calculate_path_to_nearest_coin(self, game_state, nearest_coin, features)
    features['COIN_DIRECTION'] = nearest_coin_path

    # FEATURE3: escaping bomb
    features['BOMB_ESCAPE'] = escape_bomb(self, game_state)

    # FEATURE4: CRATE
    features['CRATE_PRESENCE'] = check_the_presence_of_crates(self, game_state)
    self.logger.debug(f"crate presence: {features['CRATE_PRESENCE']}")

    features = dict(sorted(features.items()))
    self.logger.debug(f"features - state_to_features: {features}")

    # self.logger.debug(f"self_valid_actions_list: {self.valid_actions_list}")

    for i, action in enumerate(self.valid_actions_list):
        if action == features:
            self.logger.info(f"Action from state_to_feature = {i}: {action}")
            return i

    # return int(obstacle_state)


def find_missing_directions(self, directions) -> np.array:
    # directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    direction_set = set(directions)
    action_set = set(ACTIONS)

    # Find the difference between the two sets, and remove 'WAIT' if present
    missing_direction = list(action_set - direction_set)
    return [dir for dir in missing_direction if dir != 'WAIT' and dir != 'NO_OBSTACLE' and dir != 'BOMB']


def load_q_table(self, q_table_folder):
    try:
        file = os.listdir(q_table_folder)

        # getting file path
        q_table_file = q_table_folder + file[1]
        # self.logger.info(f"Loading Q-table from {q_table_file}")
        q_table = np.load(q_table_file)
        # self.logger.info("Loading Q-table from saved state.")
        return q_table
    except FileNotFoundError:
        self.logger.info("No Q-table found")
        return None


def check_the_presence_of_walls(self, game_state: dict) -> np.array:
    arena = game_state['field']
    current_position = game_state['self'][-1]
    coord_x, coord_y = current_position
    obstacle = []

    #  because the field in the coordinates is transposed
    if arena[coord_x][coord_y + 1] == -1:
        obstacle.append('DOWN')
    if arena[coord_x][coord_y - 1] == -1:
        obstacle.append('UP')
    if arena[coord_x - 1][coord_y] == -1:
        obstacle.append('LEFT')
    if arena[coord_x + 1][coord_y] == -1:
        obstacle.append('RIGHT')
    else:
        obstacle.append('NO_OBSTACLE')

    return obstacle


def check_the_presence_of_crates(self, game_state: dict) -> np.array:
    arena = game_state['field']
    current_position = game_state['self'][-1]
    coord_x, coord_y = current_position
    crate = ''

    #  because the field in the coordinates is transposed
    if arena[coord_x][coord_y + 1] == 1 or arena[coord_x][coord_y - 1] == 1 or arena[coord_x - 1][coord_y] == 1 or arena[coord_x + 1][coord_y] == 1:
        crate = 'CRATE'
    else:
        crate = 'NO_CRATE'

    return crate


def check_for_coin_presence(self, game_state: dict, near_tiles=9) -> (int, list):
    """
    This function checks if the coin is present on the field within a certain radius or not.
    Args:
        self: the game state holder
        game_state: state of the current round of the game
        near_tiles: the near tiles of the current player to check for presence

    Returns: returns an integer containing the coin present on the field.

    """
    num_of_coins_on_field = len(game_state['coins'])
    coins_list = game_state['coins']
    agent_x, agent_y = game_state['self'][-1]
    nearest_coin = []
    for tile in range(1, near_tiles + 1):
        if (agent_x + tile, agent_y) in coins_list:
            nearest_coin.append((agent_x + tile, agent_y))
            break
        if (agent_x, agent_y + tile) in coins_list:
            nearest_coin.append((agent_x, agent_y + tile))
            break
        if (agent_x - tile, agent_y) in coins_list:
            nearest_coin.append((agent_x - tile, agent_y))
            break
        if (agent_x, agent_y - tile) in coins_list:
            nearest_coin.append((agent_x, agent_y - tile))
            break
    self.logger.debug(f'This is the nearest coin: {nearest_coin}')
    return num_of_coins_on_field, nearest_coin


def calculate_path_to_nearest_coin(self, game_state, nearest_coin, directions):
    # The agent's current position
    agent_start = tuple(game_state['self'][-1])

    # If there's no coin, return NO_COIN
    if not nearest_coin:
        return 'NO_COIN'

    # Bi-directional BFS
    path = bidirectional_bfs(agent_start, tuple(nearest_coin[0]), game_state)

    # in case of no valid path or we are at coin location
    if not path or len(path) < 2:
        return 'NO_COIN'

    # Next step is the first step after current position
    next_step = path[1]

    x_start, y_start = agent_start
    x_next, y_next = next_step

    # check for obstacle in coin path
    obstacle = check_the_presence_of_walls(self, game_state)

    # Determine the direction to move
    if x_next == x_start and y_next == y_start - 1:
        if directions['UP'] != 'OBSTACLE':
            return 'UP'
        else:
            return np.random.choice(['LEFT', 'RIGHT'])
        # return 'COIN_UP'
    elif x_next == x_start and y_next == y_start + 1:
        # return 'COIN_DOWN'
        if directions['DOWN'] != 'OBSTACLE':
            return 'DOWN'
        else:
            return np.random.choice(['LEFT', 'RIGHT'])
    elif x_next == x_start - 1 and y_next == y_start:
        if directions['LEFT'] != 'OBSTACLE':
            return 'LEFT'
        else:
            return np.random.choice(['UP', 'DOWN'])
        # return 'COIN_LEFT'
    elif x_next == x_start + 1 and y_next == y_start:
        if directions['RIGHT'] != 'OBSTACLE':
            return 'RIGHT'
        else:
            return np.random.choice(['UP', 'DOWN'])
        # return 'COIN_RIGHT'
    # IDEA: return not the coin location but the direction I should take to reach that coin
    # IDEA: if there is an obstacle in the path then return a random direction which is obstacle free
    # IDEA: if the coin is present in vertical direction and an obstacle is there then return a random horizontal direction
    # IDEA: and vice versa
    return 'NO_COIN'  # Default in case something went wrong


# MARK: BI-Directional BFS implementation

def get_neighbors_old(position, field):
    x, y = position
    neighbors = []

    # Check the possible moves (UP, DOWN, LEFT, RIGHT)
    possible_moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    # Only allow valid moves (within bounds and free tiles)
    for nx, ny in possible_moves:
        if field.shape[0] > nx >= 0 == field[nx, ny] and 0 <= ny < field.shape[1]:
            neighbors.append((nx, ny))

    return neighbors


def get_neighbors(position, field):
    x, y = position
    neighbors = []

    # Check the possible moves (UP, DOWN, LEFT, RIGHT)
    possible_moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    # Only allow valid moves (within bounds and free tiles)
    for nx, ny in possible_moves:
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:  # Check within bounds
            if field[nx, ny] == 0:  # Check if the tile is free (not an obstacle)
                neighbors.append((nx, ny))

    return neighbors


def merge_paths(meeting_node, visited_start, visited_target):
    path_start = []
    path_target = []

    # Build path from start to meeting point
    node = meeting_node
    while node:
        path_start.append(node)
        node = visited_start[node]

    # Build path from target (coin) to meeting point
    node = visited_target[meeting_node]
    while node:
        path_target.append(node)
        node = visited_target[node]

    # Combine the paths
    return path_start[::-1] + path_target


# Bidirectional BFS search
def bidirectional_bfs(start, target, game_state):
    queue_start = deque([start])
    queue_target = deque([target])
    visited_start = {start: None}
    visited_target = {target: None}

    while queue_start and queue_target:
        if queue_start:
            node = queue_start.popleft()
            if node in visited_target:
                return merge_paths(node, visited_start, visited_target)  # Join the two paths

            for neighbor in get_neighbors(node, game_state['field']):
                if neighbor not in visited_start:
                    queue_start.append(neighbor)
                    visited_start[neighbor] = node

        if queue_target:
            node = queue_target.popleft()
            if node in visited_start:
                return merge_paths(node, visited_start, visited_target)

            for neighbor in get_neighbors(node, game_state['field']):
                if neighbor not in visited_target:
                    queue_target.append(neighbor)
                    visited_target[neighbor] = node
    return None


# MARK: Explosion escape feature
# This function calculates safe and dangerous tiles for our agent
def calculate_safe_and_dangerous_tiles(explosion_map: np.array):
    # Initialize lists for safe and dangerous tiles
    safe_tiles = []
    dangerous_tiles = []

    # Iterate over the explosion map
    for x in range(explosion_map.shape[0]):
        for y in range(explosion_map.shape[1]):
            if explosion_map[x, y] == 0:
                safe_tiles.append((x, y))
            else:
                dangerous_tiles.append((x, y))

    return safe_tiles, dangerous_tiles


def is_bomb_nearby(bombs, agent_position):
    for bomb_pos, _ in bombs:
        # Check if the bomb is within 3 tiles in horizontal or vertical direction
        if abs(bomb_pos[0] - agent_position[0]) <= 3 and bomb_pos[1] == agent_position[1]:
            return True
        if abs(bomb_pos[1] - agent_position[1]) <= 3 and bomb_pos[0] == agent_position[0]:
            return True
    return False


def escape_bomb(self, game_state):
    agent_position = game_state['self'][-1]
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    # Check if there's a bomb near the agent
    if is_bomb_nearby(bombs, agent_position):
        # Find all safe tiles
        safe_tiles, _ = calculate_safe_and_dangerous_tiles(explosion_map)

        # Try to find a safe path using bidirectional BFS
        for safe_tile in safe_tiles:
            path = bidirectional_bfs(agent_position, safe_tile, game_state)
            if path:
                # Determine the direction to move towards the safe tile
                next_step = path[1]  # The first step in the path after the agent's position
                move_direction = (next_step[0] - agent_position[0], next_step[1] - agent_position[1])

                # Map the movement to an action
                for action, move in BOMB_MOVES.items():
                    if move == move_direction:
                        self.logger.debug(f'action returned by escape bomb: {action}')
                        return action

        # If no safe path is found, fallback to random valid action
        valid_actions = get_valid_actions(agent_position, game_state)
        if valid_actions:
            self.logger.debug(f'No safe path is found')
            return random.choice(valid_actions)

    # Default action if no bomb is nearby or no valid moves
    return 'SAFE'


def get_valid_actions(agent_position, game_state):
    """
    Get valid actions the agent can take based on the game field.
    This ensures that the agent doesn't try to move into obstacles or walls.
    Not using the check_for_wall_presence as to not interfere with the logic of NO_OBSTACLE and other
    features.
    """
    field = game_state['field']
    valid_actions = []

    for action, move in BOMB_MOVES.items():
        next_pos = (agent_position[0] + move[0], agent_position[1] + move[1])

        # Check if the next position is within bounds and not blocked
        if 0 <= next_pos[0] < field.shape[0] and 0 <= next_pos[1] < field.shape[1]:
            if field[next_pos[0], next_pos[1]] == 0:  # Check for free tiles (not an obstacle)
                valid_actions.append(action)

    return valid_actions
