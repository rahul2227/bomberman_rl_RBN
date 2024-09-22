import os
import random
from collections import deque

import numpy as np

from agent_code.Q_learner.helpers import ACTIONS, valid_action


def setup(self):
    q_table_folder = "q_tables/"
    self.new_state = None
    self.old_state = None
    self.valid_actions_list = valid_action()
    if self.train:
        self.logger.info("Q-learning agent from scratch")
        self.number_of_actions = len(self.valid_actions_list)
        self.Q_table = np.zeros(shape=(self.number_of_actions, len(ACTIONS)))  # currently 4 * 6
        # self.logger.info(f"Initialized Q-table = {self.Q_table}")
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
        self.exploration_rate = 0.8  # random choice
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

    # checking for obstacles
    # this returns a list of walls in the agent's vicinity
    obstacles = check_the_presence_of_walls(self, game_state)
    for direction in obstacles:
        if direction in ACTIONS:
            features[direction] = 'OBSTACLE'

    # for the remaining direction
    directions = find_missing_directions(self, obstacles)
    for direction in directions:
        features[direction] = f'MOVE_{direction}'

    coins_present, nearest_coin = check_for_coin_presence(self, game_state)

    nearest_coin_path = calculate_path_to_nearest_coin(self, game_state, nearest_coin)
    features['COIN_DIRECTION'] = nearest_coin_path

    features = dict(sorted(features.items()))
    # self.logger.debug(f"features - 73: {features}")

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
    return [dir for dir in missing_direction if dir != 'WAIT' and dir != 'NO_OBSTACLE']


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

    vicinity_coordinates = []
    if arena[coord_x][coord_y + 1] == -1:
        obstacle.append('UP')
    if arena[coord_x][coord_y - 1] == -1:
        obstacle.append('DOWN')
    if arena[coord_x - 1][coord_y] == -1:
        obstacle.append('LEFT')
    if arena[coord_x + 1][coord_y] == -1:
        obstacle.append('RIGHT')
    else:
        obstacle.append('NO_OBSTACLE')

    return obstacle


def check_for_coin_presence(self, game_state: dict, near_tiles=5) -> (int, list):
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


def calculate_path_to_nearest_coin(self, game_state, nearest_coin):
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

    # Determine the direction to move
    if x_next == x_start and y_next == y_start - 1:
        return 'COIN_UP'
    elif x_next == x_start and y_next == y_start + 1:
        return 'COIN_DOWN'
    elif x_next == x_start - 1 and y_next == y_start:
        return 'COIN_LEFT'
    elif x_next == x_start + 1 and y_next == y_start:
        return 'COIN_RIGHT'

    return 'NO_COIN'  # Default in case something went wrong


# MARK: BI-Directional BFS implementation

def get_neighbors(position, field):
    x, y = position
    neighbors = []

    # Check the possible moves (UP, DOWN, LEFT, RIGHT)
    possible_moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    # Only allow valid moves (within bounds and free tiles)
    for nx, ny in possible_moves:
        if field.shape[0] > nx >= 0 == field[nx, ny] and 0 <= ny < field.shape[1]:
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
