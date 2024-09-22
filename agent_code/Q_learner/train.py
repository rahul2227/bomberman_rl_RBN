import os.path
from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features
from .helpers import OBSTACLE_HIT, OBSTACLE_AVOID, ACTIONS, MOVED_AWAY_COIN, MOVED_TO_COIN, GOOD_MOVE, BAD_MOVE

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

    # updating states for the custom events, and it works because old_state is being updated in act function
    old_state = self.old_state  # this is the index list from the last state
    new_state = state_to_features(self, new_game_state)  # this is the index list of the new state
    previous_actions = self.valid_actions_list[old_state]
    # self.logger.debug(f'old_state: {old_state}, new_state: {new_state}')

    # Getting the agent position from the last step and current step
    old_agent_position = old_game_state['self'][-1]
    new_agent_position = new_game_state['self'][-1]

    # self.logger.debug(f'old_agent_position: {old_agent_position}, new_agent_position: {new_agent_position}')

    # Idea is to check if the agent has hit the wall by checking
    # if the agent is at the same location after the change happened
    if new_agent_position == old_agent_position:
        events.append(OBSTACLE_HIT)
    else:
        events.append(OBSTACLE_AVOID)

    # IDEA: generalized penalisation for making a wrong move if the action taken by the agent is not a good action
    # IDEA: according to the previous actions list
    # IDEA: Handle the WAIT action for the good or bad action
    if self_action == 'WAIT' or self_action != previous_actions['COIN_DIRECTION'] or 'OBSTACLE' == \
            previous_actions[self_action]:
        events.append(BAD_MOVE)
    else:
        events.append(GOOD_MOVE)
    # getting the state from the state_to_feature for old and new state
    # old_state_int = state_to_features(self, old_game_state)
    # new_state_int = state_to_features(self, new_game_state)
    # replacing because old_state_int == old_state and new_state_int == new_state

    # Append event for reward if the agent chose a direction from the current coin direction
    # If moved towards coin then MOVED_TO_COIN else MOVED_AWAY_COIN
    self.logger.debug(f'previous_actions[COIN_DIRECTION]: {previous_actions['COIN_DIRECTION']}')
    self.logger.debug(f'Action taken: {self_action}')
    if previous_actions['COIN_DIRECTION'] == self_action:
        self.logger.debug(f'This is the COIN_DIRECTION FROM FEATURES: {previous_actions["COIN_DIRECTION"]}')
        self.logger.debug(f'Action taken: {self_action}')
        events.append(MOVED_TO_COIN)
    elif previous_actions['COIN_DIRECTION'] == 'NO_COIN':
        pass
    else:
        events.append(MOVED_AWAY_COIN)

    # Initializing q-table entry
    if old_state not in self.Q_table:
        self.Q_table[old_state] = np.zeros(len(ACTIONS))
    if new_state not in self.Q_table:
        self.Q_table[new_state] = np.zeros(len(ACTIONS))

    # Mapping Action to index
    # action_index = ACTIONS.index(self_action)
    if self_action in ACTIONS:
        action_index = ACTIONS.index(self_action)
    else:
        self.logger.dbeug(f"Invalid action: {self_action}")

    self.logger.debug(f'action_index: {action_index}')

    # reward calculation
    reward = reward_from_events(self, events)

    # Q-value update
    old_q_value = self.Q_table[old_state][action_index]
    future_optimal = np.argmax(self.Q_table[new_state])

    # self.logger.debug(f'future_optimal: {future_optimal}, old_q_value: {old_q_value}')

    # update Q-value
    self.Q_table[new_state][action_index] = old_q_value + self.learning_rate * (
            reward + self.discount_rate * future_optimal - old_q_value)

    # saving current state
    self.old_state = new_state

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

    # save the q-table and check for existing also
    q_table_file_path = "q_tables/q_table.npy"
    if os.path.exists(q_table_file_path):
        previous_q_table = np.load(q_table_file_path)
        previous_q_table[self.Q_table != 0] = self.Q_table[self.Q_table != 0]
        self.Q_table = previous_q_table

    np.save(q_table_file_path, self.Q_table)
    self.logger.debug(f"Q-table empty states: {np.sum(np.all(self.Q_table == 0, axis=1))}")
    self.logger.debug(f'Q-table: {self.Q_table}')
    self.logger.debug(f'Game Field: {last_game_state['field']}')

    # transition log
    self.transitions.append(
        Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get, to en/discourage a
    certain behavior.
    """
    game_rewards = {  # making the negatives to zero
        e.COIN_COLLECTED: 20,
        MOVED_TO_COIN: 30,
        MOVED_AWAY_COIN: 0,  # before -12
        OBSTACLE_HIT: 0,  # TODO: award this also when the agent has hit another agent  # -15
        OBSTACLE_AVOID: 20,
        e.KILLED_SELF: -50,
        e.BOMB_DROPPED: 0,
        e.WAITED: 0,  # -25
        GOOD_MOVE: 18,
        BAD_MOVE: 0,  # -13
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
