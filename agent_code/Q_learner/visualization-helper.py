# helper functions to read the q table and then plot the visualizations
# for its better understanding

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from agent_code.Q_learner.helpers import ACTIONS


# load the q-table
def load_table(q_table_folder):
    try:
        file = os.listdir(q_table_folder)

        # getting file path
        q_table_file = q_table_folder + file[1]
        # self.logger.info(f"Loading Q-table from {q_table_file}")
        q_table = np.load(q_table_file)
        # self.logger.info("Loading Q-table from saved state.")
        return q_table
    except FileNotFoundError:
        print("No Q-table found")
        return None


def visualize_q_table(q_table):
    # Visualizing the q-table using a heatmap.
    plt.figure(figsize=(10, 6))
    plt.imshow(q_table, cmap='viridis', aspect='auto')
    plt.colorbar(label='Q-value')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.xticks(range(q_table.shape[1]), ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'])  # , 'BOMB'
    plt.title('Q-table Heatmap')
    plt.show()


def plot_q_table_heatmap(q_table, actions):
    """
    Plots a heatmap of the Q-values for each state-action pair.

    :param q_table: 2D NumPy array where each row is a state and each column is an action.
    :param actions: List of actions corresponding to the columns of the Q-table.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(q_table, annot=True, cmap='coolwarm', xticklabels=actions)
    plt.title('Q-table Heatmap: Q-values for State-Action Pairs')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.show()


# count zero values
def count_zero_rows(q_table):
    zero_rows = np.sum(np.all(q_table == 0, axis=1))
    print("Number of zero actions in Q-table:", zero_rows)


q_table_directory = 'q_tables/'
q_table = load_table(q_table_directory)
if q_table is not None:
    print("Q-table shape", q_table.shape)
    print('loaded q-table:', q_table)
    count_zero_rows(q_table)
    plot_q_table_heatmap(q_table, ACTIONS)
    visualize_q_table(q_table)
