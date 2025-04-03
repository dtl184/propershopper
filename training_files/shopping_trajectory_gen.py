import json
import socket
import numpy as np
import os
from env_files.utils import recv_socket_data
from base_agent import BaseAgent
from planner import HyperPlanner
import argparse
import logging
from socket_env import obj_pos_dict, obj_state_index_dict
logging.basicConfig(
    filename='shopping.log',
    filemode='w',  # Overwrites the log file each time. Use 'a' to append.
    level=logging.DEBUG,  # Capture all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True  # Ensures previous configurations don’t interfere
)

def euclidean_distance(pos1, pos2):
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def calculate_reward(previous_state, current_state, target, task, index):
    reached = False
    global min_distance

    if task == 'navigate':
        prev_player = previous_state['observation']['players'][0]
        curr_player = current_state['observation']['players'][0]
        target_pos = obj_pos_dict[target]
        prev_dist = euclidean_distance(prev_player['position'], target_pos)
        curr_dist = euclidean_distance(curr_player['position'], target_pos)
        dist_gain = prev_dist - curr_dist

        # if curr_dist < min_distance:
        #     min_distance = curr_dist
        #     return reached, 10

        if index in obj_state_index_dict[target]:
            reached = True
            return reached, 100
        
        return reached, -1

    if task == 'get':
        if target == 'basket':
            if len(current_state['observation']['baskets']) != 0:
                reached = True
                return reached, 100
        else:
            if len(current_state['observation']['baskets']) == 0:
                return reached, -5
            basket_contents = current_state['observation']['baskets'][0]['contents']
            if target in basket_contents:
                target_id = basket_contents.index(target)
                if current_state['observation']['baskets'][0]['contents_quant'][target_id] > 0:
                    reached = True
                    return reached, 100

        return reached, -1
    
    if task == 'pay':
        if len(current_state['observation']['baskets']) == 0:
            return reached, -5
        if len(current_state['observation']['baskets'][0]['purchased_contents']) != 0:
            reached = True
            return reached, 100
        return reached, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument('--num_trajectories', type=int, default=10, help="Number of trajectories to generate")
    args = parser.parse_args()

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    trajectory_dir = "hybrid_shopping_trajectories/"
    qtable_dir = "hybrid_shopping_qtables/"
    os.makedirs(trajectory_dir, exist_ok=True)
    os.makedirs(qtable_dir, exist_ok=True)

    for trajectory_idx in range(args.num_trajectories):
        planner = HyperPlanner([], agent_class=BaseAgent)
        planner.load_qtables()
        planner.reset()
        sock_game.send(str.encode("0 RESET"))
        state = recv_socket_data(sock_game)
        state = json.loads(state)


        planner.parse_shopping_list(state)
        trajectory = []
        norm_violations = 0
        while not planner.plan_finished():
            agent = planner.get_agent()
  
            agent.epsilon = 0
            
            task = planner.get_task()
            task_name = task.replace(" ", "_")
            trajectory_file = os.path.join(trajectory_dir, f"trajectory_{task_name}.txt")
            
            
            qtable_path = os.path.join(qtable_dir, f"qtable_{task_name}.json")
            if os.path.exists(qtable_path):
                agent.load_qtable(qtable_path)
            else:
                continue
            
            steps = 0
            goal_reached = False
            
            
            while steps < 300 and not goal_reached:
                steps += 1
                action_index = np.argmax(agent.qtable.get(agent.state_index(state), np.zeros(len(agent.action_space)-1)))
                num = 6 if action_index <= 3 else 1
                for _ in range(num):  
                    action = "0 " + agent.action_space[action_index]
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)
                    next_state = json.loads(next_state) if next_state else None

                goal_reached, _ = calculate_reward(state,
                                        next_state,
                                        target=planner.get_task().split()[1],
                                        task=planner.get_task().split()[0],
                                        index=agent.state_index(state))
                
                norm_violations += len(state.get('violations', []))
                state = next_state
                
                if planner.get_task() not in planner.plan:
                    goal_reached = True
                    
            planner.update()
        # with open('test.txt', "a") as file:
        #     file.write(str(trajectory) + "\n")
        logging.info(f'Run {trajectory_idx} Norm Violations: {norm_violations}')

            

    
    sock_game.close()
