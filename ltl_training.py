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

def save_cumulative_rewards(cumulative_rewards, filename="cumulative_rewards_idk.csv"):
    """
    Save the cumulative rewards to a CSV file.

    Args:
        cumulative_rewards (list): List of cumulative rewards per episode.
        filename (str): File name to save the data.
    """
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        # File exists and is not empty
        existing_data = pd.read_csv(filename)
        new_data = pd.DataFrame({"Episode": range(len(existing_data) + 1, len(existing_data) + len(cumulative_rewards) + 1),
                                 "Cumulative Reward": cumulative_rewards})
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # File does not exist or is empty
        combined_data = pd.DataFrame({"Episode": range(1, len(cumulative_rewards) + 1), 
                                      "Cumulative Reward": cumulative_rewards})

    combined_data.to_csv(filename, index=False)
    print(f"Cumulative rewards saved to {filename}")

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
    parser.add_argument('--training_time', type=int, default=500, help="Number of training episodes")
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
    cumulative_rewards = []
    history = []

    for i in range(1, training_time + 1):
        print(f'Starting episode: {i}\n')
        try:
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

            while not state['gameOver']:
                cnt += 1
                agent.current_state = agent.trans(state)
                # Choose a new action only if the agent transitions to a new state
                current_state_index = agent.coords_to_state(
                    int(round(state['observation']['players'][0]['position'][0])),
                    int(round(state['observation']['players'][0]['position'][1]))
                )

                if current_state_index != last_state_index:
                    # Agent has entered a new grid square, reward it and choose a new action
                    goal_reached, reward = calculate_reward(state, state, agent.goal)
                    cur_ep_return += reward

                    # Choose the next action
                    action_index = agent.choose_action(state)
                    last_action = action_index

                    # Update Q-table for the transition
                    agent.learning(last_action, state, state, reward)
                else:
                    if np.random.uniform(0, 1) < .85:
                        action_index = last_action
                    else:
                        action_index = random.choice(range(len(agent.action_space)))

                action = "0 " + agent.action_space[action_index]
                sock_game.send(str.encode(action))

                # If still in the same state, repeat the last action


                # Update the state
                next_state = recv_socket_data(sock_game)
                next_state = safe_json_loads(next_state)

                if next_state is None:
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    history.append(0)
                    break

                state = next_state
                last_state_index = current_state_index

                if cnt > episode_length:
                    history.append(0)
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    break

                if goal_reached:  # Success condition
                    history.append(1)
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    break

            # Track success rate over the last 50 episodes
            history = history[-50:]
            if i % 100 == 0:
                print(f"Success rate: {np.mean(history)}")
        except ConnectionAbortedError as e:
            print(f"Connection error during episode {i}: {e}. Resetting for next episode.")
            try:
                sock_game.close()
                sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock_game.connect((HOST, PORT))
            except Exception as reconnect_error:
                print(f"Failed to reconnect: {reconnect_error}. Exiting.")
                break
        except json.JSONDecodeError as e:
            print(f"JSON decode error in episode {i}: {e}. Skipping to next episode.")
            history.append(0)
            continue
        except Exception as e:
            print(f"Unexpected error in episode {i}: {e}. Skipping to next episode.")
            history.append(0)
            continue

    sock_game.close()
    save_cumulative_rewards(cumulative_rewards)

if __name__ == "__main__":
    main()
