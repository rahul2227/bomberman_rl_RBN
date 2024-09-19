import os
import pickle
import random

import numpy as np

from agent_code.Q_learner.helpers import ACTIONS, action_index, valid_action


def setup(self):
    q_table_folder = "q_tables/"
    self.new_state = None
    self.old_state = None
    self.valid_actions_list = valid_action()
    if self.train:
        self.logger.info("Q-learning agent from scratch")
        self.number_of_actions = len(self.valid_actions_list)
        self.Q_table = np.zeros(shape=(self.number_of_actions, len(ACTIONS)))  # currently 4 * 6
        self.logger.info(f"Initialized Q-table = {self.Q_table}")
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
    for obstacle in obstacles:
        if obstacle in ACTIONS:
            features[obstacle] = 'OBSTACLE'
        else:
            features[obstacle] = f'MOVE_{obstacle}'
    # num_of_coins = check_for_coin_presence(self, game_state)

    # converting the features into a more indexable format for easy access in q-table
    # obstacle_state = ''.join([str(obstacle) for obstacle in obstacles])

    # feature one tells the location of the walls present near the agent
    # features['obstacles'] = int(obstacle_state)
    # features['WALL_PRESENCE'] = obstacles
    # feature for coins
    # features['coins'] = num_of_coins

    for i, action in enumerate(self.valid_actions_list):
        if action == features:
            return i

    # return int(obstacle_state)


def load_q_table(self, q_table_folder):
    try:
        file = os.listdir(q_table_folder)

        # getting file path
        q_table_file = q_table_folder + file[0]
        q_table = np.load(q_table_file)
        self.logger.info("Loading Q-table from saved state.")
        return q_table
    except FileNotFoundError:
        self.logger.info("No Q-table found")
        return None


def check_the_presence_of_walls(self, game_state: dict) -> np.array:
    arena = game_state['field']
    current_position = game_state['self'][-1]
    coord_x, coord_y = current_position
    obstacle = []

    if (coord_x + 1, coord_y) == -1:
        obstacle.append(action_index['RIGHT'])
    elif (coord_x - 1, coord_y) == -1:
        obstacle.append(action_index['LEFT'])
    elif (coord_x, coord_y + 1) == -1:
        obstacle.append(action_index['UP'])
    elif (coord_x, coord_y - 1) == -1:
        obstacle.append(action_index['DOWN'])
    else:
        obstacle.append(action_index['NO_OBSTACLE'])

    return obstacle


def check_for_coin_presence(self, game_state: dict) -> int:
    """
    This function checks if the coin is present on the field or not.
    Args:
        self: the game state holder
        game_state: state of the current round of the game

    Returns: returns an integer containing the coin present on the field.

    """
    num_of_coins = len(game_state['coins'])
    return num_of_coins
