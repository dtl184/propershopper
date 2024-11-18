import argparse
import socket
import json
from irl_agent import IRLAgent

# Placeholder for a function to receive socket data
def recv_socket_data(sock):

    data = sock.recv(4096).decode()  # Adjust buffer size if necessary
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        default=9000,
        help="Port to connect to the environment"
    )
    args = parser.parse_args()

    # Connect to the environment
    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    trajectories = [
        {"state": "A", "action": "NORTH", "next_state": "B"},
        {"state": "B", "action": "EAST", "next_state": "C"},
        {"state": "C", "action": "INTERACT", "next_state": "D"},
    ]

    agent = IRLAgent(trajectories)

    sock_game.send(str.encode("0 RESET"))  
    state = recv_socket_data(sock_game)
    state = json.loads(state)

    done = False
    while not done:
        # Agent chooses the next action based on the state
        action_index = agent.choose_action(state)
        action = "0 " + agent.action_commands[action_index]
        print(f"Sending action: {action}")

        # Send the action to the environment
        sock_game.send(str.encode(action))
        next_state = recv_socket_data(sock_game)
        next_state = json.loads(next_state)

        # Check if the basket is picked up (example condition)
        if "basket" in next_state.get("inventory", []):
            print("Basket picked up! Task complete.")
            done = True

        # Update the state
        state = next_state

    sock_game.close()
    print("Test complete.")

if __name__ == "__main__":
    main()
