import subprocess
import time


def run_script_in_new_terminal(script_name, sleep_time = 0):
    """Run a Python script in a new terminal window."""
    try:
        if sleep_time > 0:
            time.sleep(sleep_time)
        # Command to open a new terminal window and run the script
        terminal_command = ['gnome-terminal', '--', 'bash', '-c', f'python3 {script_name}; exec bash']
        # Start the new terminal and run the script
        subprocess.Popen(terminal_command)
        print(f"Running {script_name} in a new terminal.")
    except Exception as e:
        print(f"Failed to start {script_name} in a new terminal: {e}")

def main():
    # List of scripts to run
    scripts = ['socket_env.py --keyboard --num_player=4 --mode=999 --stochastic=True', "astar_path_planner_0.py", 
               "astar_path_planner_1.py", "astar_path_planner_2.py", "astar_path_planner_3.py" ]
    sleep_time = 0.5 * len(scripts)
    cnt = 0
    # Run each script in a new terminal window
    for script in scripts:
        run_script_in_new_terminal(script)
        sleep_time -= 0.5
        if cnt == 0:
            time.sleep(2)

if __name__ == "__main__":
    main()
