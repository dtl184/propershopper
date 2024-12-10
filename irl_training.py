import argparse
import socket
import json
from irl_agent import IRLAgent
import numpy as np
from utils import recv_socket_data
import matplotlib.pyplot as plt


def load_trajectories(file_name):
    """
    Reads trajectories from a text file and stores them as a list of lists.

    Parameters:
        file_name (str): The name of the file containing the trajectories.

    Returns:
        list: A list where each trajectory is a separate list of tuples.
    """
    trajectories = []

    with open(file_name, "r") as file:
        for line in file:
            # Strip whitespace and newlines
            line = line.strip()
            if line:  # Ensure the line is not empty
                # Evaluate the line as a Python list
                trajectory = eval(line)
                trajectories.append(trajectory)

    return trajectories

def pad_trajectories(file_path):
    """
    Reads trajectories from a text file, pads each trajectory to the length
    of the longest trajectory by repeating the last element, and writes the
    padded trajectories to a new file.

    Parameters:
        file_path (str): The path to the input file.
        output_path (str): The path to save the output file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert each line to a list of tuples
    trajectories = [eval(line.strip()) for line in lines]

    # Find the maximum length of trajectories
    max_length = max(len(traj) for traj in trajectories)

    # Pad each trajectory to the maximum length
    padded_trajectories = []
    for traj in trajectories:
        if len(traj) < max_length:
            last_element = traj[-1]
            traj += [last_element] * (max_length - len(traj))
        padded_trajectories.append(traj)

    return padded_trajectories



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        default=9000,
        help="Port to connect to the environment"
    )
    args = parser.parse_args()

    #trajectories = load_trajectories('trajectories.txt')

    trajectories = pad_trajectories("trajectories.txt")

    agent = IRLAgent(n_states=437, trajectories=trajectories)

    with open("learned_reward.txt", "r") as file:
         agent.set_reward(np.array(eval(file.read())))

    agent.learn_reward()

    plt.subplot(1, 2, 2)
    plt.pcolor(agent.reward.reshape((19, 23)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    sock_game.send(str.encode("0 RESET"))  
    state = recv_socket_data(sock_game)
    state = json.loads(state)

    done = False
    while not done:
        # Agent chooses the next action based on the state
        x = int(round(state['observation']['players'][0]['position'][0]))
        y = int(round(state['observation']['players'][0]['position'][1]))
        action_index = agent.choose_action((x, y))
        action = "0 " + agent.action_space[action_index]
        print(f"Sending action: {action}")

        # Send the action to the environment
        sock_game.send(str.encode(action))
        next_state = recv_socket_data(sock_game)
        next_state = json.loads(next_state)

        # Check if the basket is picked up (example condition)
        # if "basket" in next_state.get("inventory", []):
        #     print("Basket picked up! Task complete.")
        #     done = True

        # Update the state
        state = next_state

    sock_game.close()
    print("Test complete.")

if __name__ == "__main__":
    main()
