import argparse
import socket
import json
from ltl_agent import LTLAgent
import numpy as np
from utils import recv_socket_data
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

def calculate_reward(state, next_state, goal=None):
    global min_distance  # Track the minimum distance to the goal
    # Default reward
    reward = -1

    min_distance = 100 # start with large val

    reached = False
    
    if goal is not None:
        curr_position = state['observation']['players'][0]['position']
        next_position = next_state['observation']['players'][0]['position']
        
        # calculate the agent's distance to the goal from the next state
        curr_distance = ((curr_position[0] - goal[0]) ** 2 + (curr_position[1] - goal[1]) ** 2) ** 0.5
        next_distance = ((next_position[0] - goal[0]) ** 2 + (next_position[1] - goal[1]) ** 2) ** 0.5
        #print(f'distance to goal: {distance}\n')
        
        # reward the agent more the closer it gets to the goal
        # if next_distance < min_distance:
        #     min_distance = next_distance:
        #     reward = 10
        
        # Check if the goal has been reached
        if curr_distance < 1:  # Tolerance for reaching the goal
            reward = 1000
            reached = True
            print("Goal reached!")
        else:
            reward = (next_distance - curr_distance) * 5 - 0.05 - 20*len(next_state['violations'])
            

    return reached, reward

def load_trajectories(file_name):
    trajectories = []
    with open(file_name, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                trajectory = eval(line)
                trajectories.append(trajectory)
    return trajectories

def safe_json_loads(data):
    """
    Safely loads JSON data, with error handling for empty or invalid data.
    
    Args:
        data (str): The JSON string to parse.
        
    Returns:
        dict: Parsed JSON data, or an empty dictionary if parsing fails.
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None

import pandas as pd

import pandas as pd
import ast

def save_experiment_rewards(current_experiment, current_episode_reward, filename="experiment_rewards.csv"):
    """
    Appends the current episode reward to the row corresponding to the current experiment.
    If the experiment row doesn't exist, it creates one.
    
    Parameters:
    - current_experiment (int): The current experiment number.
    - current_episode_reward (float): The reward for the current episode.
    - filename (str): The name of the CSV file to save the rewards.
    """
    # Check if the file exists and load it, or create a new DataFrame
    try:
        df = pd.read_csv(filename)
        # Convert the "Rewards" column back to a list if it exists
        if "Rewards" in df.columns:
            df["Rewards"] = df["Rewards"].apply(ast.literal_eval)
    except (FileNotFoundError, ValueError):
        # Create a new DataFrame if the file doesn't exist or is empty
        df = pd.DataFrame(columns=["Experiment Number", "Rewards"])

    # Check if the current experiment exists in the DataFrame
    if current_experiment in df["Experiment Number"].values:
        # Append the reward to the existing rewards list
        df.loc[df["Experiment Number"] == current_experiment, "Rewards"].iloc[0].append(current_episode_reward)
    else:
        # Create a new row for the experiment
        new_row = {"Experiment Number": current_experiment, "Rewards": [current_episode_reward]}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)





def smooth_rewards(rewards, window_size=5):
    return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

def plot_smoothed_rewards(filename="cumulative_rewards.csv", window_size=5):
    """
    Plot the smoothed rewards from the CSV file using a moving average.

    Args:
        filename (str): The CSV file containing rewards data.
        window_size (int): The size of the moving average window for smoothing.
    """
    df = pd.read_csv(filename)

    # Apply moving average for smoothing
    df['Smoothed Reward'] = df["Cumulative Reward"].rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df["Episode"], df["Cumulative Reward"], alpha=0.5, label="Raw Reward")
    plt.plot(df["Episode"], df["Smoothed Reward"], label=f"Smoothed Reward (window={window_size})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Training Episodes (Smoothed)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_average_reward(filename="cumulative_rewards.csv", window_size=5):
    """
    Plot the average reward over a specified window size.

    Args:
        filename (str): CSV file containing cumulative rewards data with columns 'Episode' and 'Cumulative Reward'.
        window_size (int): Number of episodes to average over. Default is 5.
    """
    # Load the data
    df = pd.read_csv(filename)

    # Calculate the moving average
    df['Average Reward'] = df['Cumulative Reward'].rolling(window=window_size).mean()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Episode'], df['Average Reward'], label=f"Average Reward (Window={window_size})", color="blue", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Episodes (Scatter Plot)")
    plt.legend()
    plt.grid()
    plt.show()
    


def plot_normalized_rewards(filename="cumulative_rewards_idk.csv"):
    """
    Plot the normalized rewards from the CSV file.
    """
    df = pd.read_csv(filename)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Episode"], df["Cumulative Reward"], label="Normalized Reward per Step")
    plt.xlabel("Episode")
    plt.ylabel("Normalized Reward")
    plt.title("Normalized Reward Over Training Episodes")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument('--training_time', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--episode_length', type=int, default=500, help="Maximum steps per episode")
    args = parser.parse_args()

    trajectories = load_trajectories('trajectories.txt')
    agent = LTLAgent(n_states=437, goal=(3, 18))
    agent.learn_from_trajectories(trajectories)
    #plot_normalized_rewards()
    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = args.training_time
    episode_length = args.episode_length
    
    num_experiments = 10
    experiment_rewards = []
    # Whole training loop
    for experiment in range(num_experiments + 1):
        # reset q tables at beginning of each experiment
        agent.qtable = pd.DataFrame(columns=[i for i in range(len(agent.action_space))])
        agent.epsilon = 0.8 # reset epsilon because of decay
        cumulative_rewards = [] # hol
        # Experiment loop
        for i in range(1, training_time + 1):
            history = []

            #print(f'Starting episode: {i}\n')
            sock_game.send(str.encode("0 RESET"))  # reset the game
            state = recv_socket_data(sock_game)
            state = safe_json_loads(state)

            if state is None:
                history.append(0)
                continue

            cnt = 0
            cur_ep_return = 0.0
            last_state_index = 0
            last_action = 0

            # Episode loop
            while not state['gameOver']:
                foo = 0
                cnt += 1
                agent.current_state = agent.trans(state)
                # Choose a new action only if the agent transitions to a new state
                current_state_index = agent.coords_to_state(
                    int(round(state['observation']['players'][0]['position'][0])),
                    int(round(state['observation']['players'][0]['position'][1]))
                )
                action_index = agent.choose_action(state)
                while foo < 5:
                    foo += 1
                    action = "0 " + agent.action_space[action_index]
                    sock_game.send(str.encode(action))

                    # Update the state
                    next_state = recv_socket_data(sock_game)
                    next_state = safe_json_loads(next_state)

                goal_reached, reward = calculate_reward(state, next_state, agent.goal)
                cur_ep_return += reward

                agent.learning(action_index, state, next_state, reward)

                norms = next_state["violations"]

                if next_state['observation']['players'][0]['position'][0] < 0:
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    break

                if norms != '' and norms[0] == 'Player 0 exited through an entrance':
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    break

                if next_state is None or cnt > episode_length or goal_reached or next_state["gameOver"]:
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    if goal_reached:
                        history.append(1)
                    else:
                        history.append(0)
                    break

                state = next_state

            # episode finished
            print(f'Experiment {experiment} episode {i} reward: {cur_ep_return / cnt}')
            #experiment_rewards.append(cumulative_rewards)
            save_experiment_rewards(experiment, cur_ep_return / cnt)
            agent.save_qtable()
        

        # then begin a new experiment with new q table

    sock_game.close()
    save_experiment_rewards(cumulative_rewards)

if __name__ == "__main__":
    main()
