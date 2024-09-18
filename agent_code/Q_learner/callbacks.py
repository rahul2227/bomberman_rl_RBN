import os
import pickle
import random

import numpy as np

from agent_code.Q_learner.helpers import WALL_RIGHT, WALL_UP, WALL_DOWN, WALL_LEFT, ACTIONS


def setup(self):
    q_table_folder = "q_tables/"
    self.new_state = None
    self.old_state = None
    self.valid_state_list = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    if self.train:
        self.logger.info("Q-learning agent from scratch")
        self.number_of_states = len(self.valid_state_list)
        self.Q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))  # currently 4 * 6
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
        self.exploration_rate = 0.2  #random choice
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
    if self.train and random.random() < self.exploration_rate:
        self.logger.debug("Choosing action purely at random.")
        # taking a random action for exploration
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    # Querying the Q-table for an action
    print(np.argmax(self.Q_table[self.old_state]))
    action = np.random.choice(ACTIONS)
    # action = ACTIONS[np.argmax(self.Q_table[self.old_state])]
    return action


def state_to_features(self, game_state: dict) -> np.array:
    features = {}

    # checking for obstacles
    obstacles = check_the_presence_of_obstacles(self, game_state)

    # feature one tells the location of the walls present near the agent
    features['obstacles'] = obstacles

    # converting the features into a more indexable format for easy access in q-table
    obstacle_state = ''.join(sorted([list(obstacle.keys())[0] for obstacle in obstacles])) or "NO_OBSTACLE"

    return obstacle_state


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


def check_the_presence_of_obstacles(self, game_state: dict) -> np.array:
    arena = game_state['field']
    current_position = game_state['self'][-1]
    coord_x, coord_y = current_position
    obstacle = []
    wall_presence = {}

    if arena[coord_x + 1][coord_y] == 1 or (coord_x + 1, coord_y) == -1:
        obstacle.append({'WALL_RIGHT': WALL_RIGHT})  # this will add Right wall presence in the list
    elif arena[coord_x - 1][coord_y] == 1 or (coord_x - 1, coord_y) == -1:
        obstacle.append({'WALL_LEFT': WALL_LEFT})
    elif arena[coord_x][coord_y + 1] == 1 or (coord_x, coord_y + 1) == -1:
        obstacle.append({'WALL_UP': WALL_UP})
    elif arena[coord_x][coord_y - 1] == 1 or (coord_x, coord_y - 1) == -1:
        obstacle.append({'WALL_DOWN': WALL_DOWN})

    return obstacle
