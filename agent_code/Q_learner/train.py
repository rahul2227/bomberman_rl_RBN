import os.path
from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features
from .helpers import OBSTACLE_HIT, OBSTACLE_AVOID, ACTIONS

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters
TRANSITION_HISTORY_SIZE = 3
RECORD_ENEMY_TRANSITIONS = 1.0

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    self.discount_rate = 0.2  # γ
    self.learning_rate = 0.1  # α
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Getting the agent position from the last step and current step
    old_agent_position = old_game_state['self'][-1]
    new_agent_position = new_game_state['self'][-1]

    # Idea is to check if the agent has hit the wall by checking
    # if the agent is at the same location after the change happened
    if new_agent_position == old_agent_position:
        events.append(OBSTACLE_HIT)
    else:
        events.append(OBSTACLE_AVOID)

    # getting the state from the state_to_feature for old and new state
    old_state_int = state_to_features(self, old_game_state)
    new_state_int = state_to_features(self, new_game_state)

    # Initializing q-table entry
    if old_state_int not in self.Q_table:
        self.Q_table[old_state_int] = np.zeros(len(ACTIONS))
    if new_state_int not in self.Q_table:
        self.Q_table[new_state_int] = np.zeros(len(ACTIONS))

    # Mapping Action to index
    action_index = ACTIONS.index(self_action)

    # reward calculation
    reward = reward_from_events(self, events)

    # Q-value update
    old_q_value = self.Q_table[old_state_int][action_index]
    future_optimal = np.argmax(self.Q_table[new_state_int])

    # update Q-value
    self.Q_table[new_state_int][action_index] = old_q_value + self.learning_rate * (reward + self.discount_rate * future_optimal - old_q_value)

    # saving current state
    self.old_state = new_state_int

    # transition logging
    self.transitions.append(
        Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Idea: Get Last state and action and update it in the q-table
    last_state_int = state_to_features(self, last_game_state)
    action_index = ACTIONS.index(last_action)

    # reward calculation
    reward = reward_from_events(self, events)

    # last game state in q-table
    if last_state_int not in self.Q_table:
        self.Q_table[last_state_int] = np.zeros(len(ACTIONS))

    # update
    old_q_value = self.Q_table[last_state_int][action_index]

    # no future state, so future_reward=0
    self.Q_table[last_state_int][action_index] = old_q_value + self.learning_rate * (reward - old_q_value)

    # save the q-table
    q_table_file_path = "q_tables/q_table.npy"
    np.save(q_table_file_path, self.Q_table)
    self.logger.info(f"Q-table saved to {q_table_file_path}")

    # transition log
    self.transitions.append(
        Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        OBSTACLE_HIT: -0.5,
        OBSTACLE_AVOID: 1,
        e.KILLED_SELF: -0.5,
        e.BOMB_DROPPED: 0,
        e.WAITED: -0.25,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
