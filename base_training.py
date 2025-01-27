import argparse
import socket
import json
import numpy as np
import pandas as pd
from base_agent import BaseAgent
from utils import recv_socket_data
import matplotlib.pyplot as plt


def calculate_reward(state, next_state, goal=None):
    """
    Calculate reward based on the distance to the goal.
    """
 
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
        
        # Check if the goal has been reached
        if curr_distance < 1:  # Tolerance for reaching the goal
            reward = 1000
            reached = True
            print("Goal reached!")
        else:
            reward = (next_distance - curr_distance) * 5 - 0.05 # action cost
            

    return reached, reward


def save_experiment_rewards(experiment_rewards, filename="experiment_rewards.csv"):
    """
    Save experiment rewards to a CSV file with two columns:
    - Experiment Number
    - Rewards (List of rewards per episode for each experiment)
    """
    # Prepare data for saving: list of experiment numbers and corresponding rewards
    experiment_data = []
    for i, rewards in enumerate(experiment_rewards, start=1):
        experiment_data.append([i, rewards])  # [Experiment Number, List of rewards]

    # Convert to DataFrame
    df = pd.DataFrame(experiment_data, columns=["Experiment Number", "Rewards"])

    # Save to CSV
    df.to_csv(filename, index=False)


def plot_cumulative_rewards(filename="new_cum.csv"):
    """
    Plot cumulative rewards from the CSV file.
    """
    rewards_df = pd.read_csv(filename)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_df["Episode"], rewards_df["Cumulative Reward"], label="Cumulative Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Training Episodes")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument("--training_time", type=int, default=1200, help="Number of training episodes")
    parser.add_argument("--episode_length", type=int, default=500, help="Maximum steps per episode")
    args = parser.parse_args()

    agent = BaseAgent(goal = (3, 18))
    

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    experiment_rewards = []
    num_experiments = 10
    # Whole training loop
    for experiment in range(1, num_experiments + 1):
        agent = BaseAgent(goal = (3, 18)) # reset q tables
        print(f"Starting experiment {experiment}")
        cumulative_rewards = []
        history = []
        # Experiment loop
        for i in range(1, args.training_time + 1):
            print(f"Starting episode {i}")

            sock_game.send(str.encode("0 RESET"))
            state = recv_socket_data(sock_game)
            state = json.loads(state)

            cnt = 0
            cur_ep_return = 0

            # Episode loop
            while not state['gameOver']:
                cnt += 1

                action_index = agent.choose_action(state)
                action = "0 " + agent.action_space[action_index]

                sock_game.send(str.encode(action))  # send action to env

                next_state = recv_socket_data(sock_game)
                next_state = json.loads(next_state)

                if next_state == None:
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    history.append(0)
                    break
                

                goal_reached, reward = calculate_reward(state, next_state, agent.goal)
                cur_ep_return += reward

                agent.learning(action_index, state, next_state, reward)
                state = next_state

                if cnt > 500:
                    history.append(0)
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    break

                if goal_reached:  # Success condition
                    history.append(1)
                    normalized_reward = cur_ep_return / cnt if cnt > 0 else 0
                    cumulative_rewards.append(normalized_reward)
                    break

                if next_state["gameOver"]:
                    break

            cumulative_rewards.append(cur_ep_return / cnt)
            print(f"Episode {i} Reward: {cur_ep_return / cnt}")

        # once experiment has concluded, save rewards
        experiment_rewards.append(cumulative_rewards)
        save_experiment_rewards(experiment_rewards)


    sock_game.close()
    agent.save_qtable()



if __name__ == "__main__":
    main()
