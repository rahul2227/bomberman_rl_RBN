import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    q_table_folder = "q_tables/"
    self.new_state = None
    self.old_state = None
    self.valid_state_list = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    if self.train:
        self.logger.info("Q-learning agent from scratch")
        self.number_of_states = len(self.valid_state_list)
        self.Q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS))) # currently 4 * 6
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
        self.exploration_rate = 0.6 #random choice
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # taking a random action for exploration
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    # TODO: here I need to return and answer from the q-table
    return np.random.choice(ACTIONS)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # TODO: Create a feature that tells you the presence of walls and its direction near the agent
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
