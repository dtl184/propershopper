import argparse
import socket
import json
import numpy as np
import matplotlib.pyplot as plt
from ltl_agent import LTLAgent
from utils import recv_socket_data


def load_trajectories(file_name):
    """
    Load trajectories from a text file.
    """
    trajectories = []
    with open(file_name, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                trajectory = eval(line)
                trajectories.append(trajectory)
    return trajectories


def save_norm_violation_data(results, filename="norm_violation_data.txt"):
    """
    Save norm violation data to a text file.
    Each row represents an experiment, and each column is the number of norm violations per episode.
    """
    np.savetxt(filename, results, fmt="%d")


def plot_average_norm_violations(results):
    """
    Plot the average number of norm violations per episode across experiments.
    """
    avg_violations = np.mean(results, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_violations, label="Average Norm Violations per Episode", linewidth=2, color='red')
    plt.xlabel("Episode")
    plt.ylabel("Average Number of Norm Violations")
    plt.title("Average Norm Violations per Episode Across Experiments")
    plt.legend()
    plt.grid()
    plt.savefig('norm_violation_data_graph.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument('--episode_length', type=int, default=100, help="Maximum steps per episode")
    parser.add_argument('--num_experiments', type=int, default=1000, help="Number of experiments to run")
    parser.add_argument('--num_episodes', type=int, default=1500, help="Number of episodes per experiment")
    args = parser.parse_args()

    trajectories = load_trajectories('base_trajectories.txt')

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    all_experiment_results = np.zeros((args.num_experiments, args.num_episodes), dtype=int)  # Store all results

    for experiment in range(args.num_experiments):
        print(f"\nStarting Experiment {experiment + 1}/{args.num_experiments}")

        # Reset agent for each experiment
        agent = LTLAgent(n_states=437, goal=(3, 18))
        agent.learn_from_trajectories(trajectories)

        for episode in range(args.num_episodes):
            sock_game.send(str.encode("0 RESET"))  # Reset environment
            state = recv_socket_data(sock_game)
            state = json.loads(state) if state else None

            if state is None:
                all_experiment_results[experiment, episode] = 0
                continue

            cnt = 0
            norm_violations = 0

            while not state['gameOver']:
                cnt += 1
                action_index = agent.choose_action(state)

                for _ in range(6):  # Take repeated actions
                    action = "0 " + agent.action_space[action_index]
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)
                    next_state = json.loads(next_state) if next_state else None

                if next_state is None:
                    break

                # Count norm violations
                norm_violations += len(state['violations'])

                if cnt >= args.episode_length or next_state["gameOver"]:
                    break

                state = next_state

            all_experiment_results[experiment, episode] = norm_violations
            print(f"Experiment {experiment + 1}, Episode {episode + 1}, Norm Violations: {norm_violations}")

    sock_game.close()

    # Save and plot results
    save_norm_violation_data(all_experiment_results)
    plot_average_norm_violations(all_experiment_results)


if __name__ == "__main__":
    main()
