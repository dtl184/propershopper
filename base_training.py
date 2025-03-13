import argparse
import socket
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from base_agent import BaseAgent  # ✅ Replaced LTLAgent with BaseAgent
from utils import recv_socket_data
import os

def calculate_reward(state, next_state, goal=(3, 18), agent=None):
    global min_distance
    reward = -1
    reached = False

    if agent is None:
        return False, 0

    curr_position = state['observation']['players'][0]['position']
    next_position = next_state['observation']['players'][0]['position']

    curr_distance = ((round(curr_position[0]) - goal[0]) ** 2 + (round(curr_position[1]) - goal[1]) ** 2) ** 0.5
    next_distance = ((round(next_position[0]) - goal[0]) ** 2 + (round(next_position[1]) - goal[1]) ** 2) ** 0.5

    if next_distance < min_distance:
        min_distance = next_distance
        reward = 10

    if agent.state_index(state) in [306, 307, 325, 326]:
        reward = 1000
        reached = True

    return reached, reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument("--trajectory", type=str, default='test.txt', help="Path to the trajectory file")
    args = parser.parse_args()

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    with open(args.trajectory, 'r') as f:
        trajectories = [eval(line.strip()) for line in f]

    for trajectory in trajectories:
        agent = BaseAgent()
        sock_game.send(str.encode("0 RESET"))
        state = recv_socket_data(sock_game)
        state = json.loads(state)

        for state_index, action_index in trajectory:
          for _ in range(6):
                action = "0 " + agent.action_space[action_index]
                sock_game.send(str.encode(action))
                next_state = recv_socket_data(sock_game)
                next_state = json.loads(next_state) if next_state else None

        if next_state is None or next_state["gameOver"]:
            break
            
        state = next_state

    sock_game.close()

if __name__ == "__main__":
    main()
