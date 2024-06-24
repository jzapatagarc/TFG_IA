import os
import pickle
import torch
import gymnasium as gym
from multiprocessing import Process, Pipe
from agent import ActorCritic
from typing import List, Dict, Tuple
from torch.optim import RMSprop
import subprocess
import sys
import time
import rl_algorithms as rl
import datetime
import pandas as pd
import shutil
import re

# Nombre alumno: José David Zapata García
# Usuario campus: jzapatagarc

# Constants for the paths
GLOBAL_MODEL_PATH = '/home/david/Documents/TFG/global'
DATA_PATH = '/home/david/Documents/TFG/data'
METRICS_PATH = '/home/david/Documents/TFG/metrics'
CSV_PATH = '/home/david/Documents/TFG/csv'
OLD_CSV_PATH = '/home/david/Documents/TFG/old_csv'


def ensure_directories_exist():
    """
    Ensures that necessary directories exist for storing models and data, creating them if they do not exist.
    """
    # Create directories if they don't exist, with no error if they already exist
    os.makedirs(GLOBAL_MODEL_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)


def load_global_model(method: str, env_name: str) -> (torch.nn.Module, RMSprop):
    """
    Loads the most recent global model for a given method and environment, or initializes a new model if none exist.

    Args:
        method (str): The name of the method used as part of the model's file naming.
        env_name (str): The name of the environment to tailor the model's input and output dimensions.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer]: A tuple containing the loaded or newly created model and its optimizer.
    """
    # Ensure required directories are available
    ensure_directories_exist()

    # Initialize the environment to determine model dimensions
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Initialize the global model
    global_model = rl.ActorCritic(input_dim, output_dim)

    # Search for existing models for the specified method and environment
    model_files = [f for f in os.listdir(GLOBAL_MODEL_PATH)
                   if f.startswith(f"{method}_{env_name}")]

    if model_files:
        # Identify the most recent model file based on the modification time
        latest_model_file = max(model_files, key=lambda f: os.path.getmtime(
            os.path.join(GLOBAL_MODEL_PATH, f)))
        global_model.load_state_dict(torch.load(
            os.path.join(GLOBAL_MODEL_PATH, latest_model_file)))
        print(f"Global model loaded from {latest_model_file}")
    else:
        print("No global model found. Starting with a new model.")

    # Initialize the optimizer
    optimizer = RMSprop(global_model.parameters(), lr=0.1)

    return global_model, optimizer


def should_process_file(filename: str, method: str, env: str) -> bool:
    """
    Determines whether a file should be processed based on the specified method and environment.

    Args:
        filename (str): The name of the file to check.
        method (str): The method used in the training.
        env (str): The environment name associated with the file.

    Returns:
        bool: True if the file matches the method and environment, False otherwise.
    """
    pattern = f"^{re.escape(method)}_{re.escape(env)}_metrics_.*\\.pkl$"
    return re.match(pattern, filename) is not None


def update_csv_with_data(method: str, env_name: str, num_agents: int):
    """
    Updates CSV files with data from processed files, organizing them by method, environment, and number of agents.

    This function collects metric data from .pkl files, aggregates them into a DataFrame, and then appends or creates
    CSV files based on the method, environment, and number of agents. Processed files are then archived.

    Args:
        method (str): Training method used.
        env_name (str): Environment name.
        num_agents (int): Number of agents involved in the data generation.
    """
    # List all .pkl files and sort them by modification time for processing in order
    files = [f for f in os.listdir(METRICS_PATH) if f.endswith('.pkl')]
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(
        os.path.join(METRICS_PATH, x)))

    data_frames: Dict[str, pd.DataFrame] = {}

    # Iterate over each file
    for file in sorted_files:
        # Check if the file matches the criteria for processing
        if should_process_file(file, method, env_name):
            full_path = os.path.join(METRICS_PATH, file)
            # Load the data from the file
            with open(full_path, 'rb') as f:
                print("Loading data from file:", file)
                data = pickle.load(f)

            # Compose a name based on method, environment, and number of agents
            key = f"{method}_{env_name}_{num_agents}_agents"
            file_path = os.path.join(CSV_PATH, f"{key}_aggregated_metrics.csv")

            # Convert loaded data to DataFrame and initialize or append to existing DataFrame
            new_data = pd.DataFrame(data, index=[0])
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                df = pd.DataFrame()

            # Concatenate new data to the existing DataFrame
            df = pd.concat([df, new_data], ignore_index=True)
            print("Appending new data to CSV for:", key)

            # Move processed file to an archive directory to avoid reprocessing
            old_file_path = os.path.join(OLD_CSV_PATH, file)
            shutil.move(full_path, old_file_path)

            # Update the data_frames dictionary
            data_frames[key] = df

    # Save updated DataFrames to CSV files
    for key, df in data_frames.items():
        file_path = os.path.join(CSV_PATH, f"{key}_aggregated_metrics.csv")
        df.to_csv(file_path, index=False)
        print(f"CSV file updated for {key} at {file_path}")


def continuous_update(method: str, env_name: str, num_agents: int):
    """
    Continuously updates CSV files with data from agents at regular intervals.

    Args:
        method (str): The training method used.
        env_name (str): The name of the environment.
        num_agents (int): Number of agents participating in the training.

    This function runs indefinitely until manually interrupted, updating CSV data files every 10 seconds.
    """
    try:
        while True:
            update_csv_with_data(method, env_name, num_agents)
            time.sleep(10)
    except KeyboardInterrupt:
        print("Update stopped manually.")


def update_model(agent_conns: List[Pipe], method_filter: str, env_name: str):
    """
    Continuously updates the global model based on the data received from agents, for 'ppo' and 'a3c' methods.

    Args:
        agent_conns (List[Pipe]): Connections to agents to send commands after updates.
        method_filter (str): Specifies the method to filter data files and manage updates.
        env_name (str): The environment name for which the model is being trained.

    The function processes incoming data files, applies updates to the global model, and notifies agents of updates.
    """
    print(f"Starting update model process for {method_filter}...")
    # Temporary storage for data until all agents have reported.
    data_storage = []

    while True:
        # List all relevant data files in the specified directory.
        data_files = [f for f in os.listdir(DATA_PATH) if f.startswith(
            method_filter) and f.endswith('.pkl')]

        for file_name in data_files:
            file_path = os.path.join(DATA_PATH, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            method = data['type']
            model_data = data['model_data']

            # Handle data according to the specified method.
            if method == 'ppo':
                data_storage.append((model_data, env_name))
            elif method == 'a3c':
                # Load the global model and apply A3C gradients immediately.
                global_model, optimizer = load_global_model(method, env_name)
                print("Applying A3C gradients.")
                rl.apply_a3c_gradients(model_data, global_model, optimizer)
                save_and_notify(global_model, method, env_name, agent_conns)

            # Remove the file after processing to prevent re-processing.
            os.remove(file_path)

        # If all data has been received for PPO, apply model updates.
        if method_filter == 'ppo' and len(data_storage) >= len(agent_conns):
            print("Received data from all agents for PPO, applying model updates...")
            for model_data, env in data_storage:
                global_model, optimizer = load_global_model(method_filter, env)
                print("Applying PPO experiences.")
                rl.apply_ppo_experiences(model_data, global_model, optimizer)
                save_and_notify(global_model, method_filter, env, agent_conns)

            # Clear the storage after processing.
            data_storage = []

        time.sleep(1)


def save_and_notify(global_model: ActorCritic, method: str, env_name: str, agent_conns: List[Pipe]):
    """
    Saves the updated global model and notifies all agents to update their local models.

    Args:
        global_model (ActorCritic): The updated global model to be saved.
        method (str): The method used, either 'ppo' or 'a3c'.
        env_name (str): The environment name associated with the model.
        agent_conns (List[Pipe]): List of connections to agents to send update commands.

    Saves the model to the GLOBAL_MODEL_PATH with a timestamp and notifies all agents.
    """
    # Save the updated model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{method}_{env_name}_{timestamp}.pth"
    torch.save(global_model.state_dict(), os.path.join(
        GLOBAL_MODEL_PATH, model_filename))
    print(f"{method.upper()} model updated and saved.")

    # Send update command to all agents
    for conn in agent_conns:
        conn.send({'command': 'update'})


def start_agent(env_name: str, method: str, max_epochs: int) -> Tuple[Process, Pipe]:
    """
    Starts an agent process to train using specified environment and method settings,
    creating a bidirectional communication channel via a Pipe.

    Args:
        env_name (str): The name of the training environment.
        method (str): The training method to use.
        max_epochs (int): The maximum number of epochs the agent should train for.

    Returns:
        Tuple[Process, Pipe]: A tuple containing the Process object of the agent and the parent connection Pipe.
    """
    # Get the directory where the current script is located.
    script_dir = os.path.dirname(__file__)
    # Path to the agent script.
    agent_script_path = os.path.join(script_dir, 'agent.py')

    # Create a Pipe for bidirectional communication.
    parent_conn, child_conn = Pipe()

    # Configure the file descriptor of the child side to be inheritable.
    fd = child_conn.fileno()
    os.set_inheritable(fd, True)

    # Start the agent process, passing the file descriptor as an argument.
    p = Process(target=invoke_agent_script, args=(
        fd, agent_script_path, env_name, method, max_epochs))
    p.start()

    return p, parent_conn


def invoke_agent_script(conn_fd: int, agent_script_path: str, env_name: str, method: str, max_epochs: int):
    """
    Invokes an agent script with necessary parameters as command-line arguments,
    including a connection file descriptor for command communication.

    Args:
        conn_fd (int): File descriptor for the connection to communicate with the parent process.
        agent_script_path (str): File path to the agent script.
        env_name (str): Name of the environment in which the agent will operate.
        method (str): The training method.
        max_epochs (int): Maximum number of training epochs.
    """
    cmd = [
        # Use the Python executable that's running the script
        sys.executable, agent_script_path,
        '--env_name', env_name,
        '--method', method,
        '--max_epochs', str(max_epochs),
        '--conn', str(conn_fd)
    ]
    subprocess.run(cmd, close_fds=False)


def user_interaction():
    """
    Handles user commands to manage training processes interactively. Allows starting, stopping,
    and exiting agent training sessions dynamically based on user input.

    This function interacts with the user through the console to control training processes for different
    training methods and environments. It manages processes and communication pipes to ensure coordinated
    operations between the main process and spawned agent processes.
    """
    # List to keep track of agent training processes
    agent_processes: List[Process] = []
    # Connections for sending commands to agents
    agent_conns: List[Pipe] = []

    try:
        while True:
            command = input(
                "Enter command (start/stop/exit): ").strip().lower()
            if command == "start":
                # Prompt the user for necessary details to start the training
                env_name = input("Enter environment name: ").strip()
                method = input("Enter method (ppo/a3c): ").strip().lower()
                max_epochs = int(input("Enter max epochs: ").strip())
                num_instances = int(
                    input("Enter number of instances: ").strip())

                for _ in range(num_instances):
                    # Start each agent as a separate process
                    process, conn = start_agent(env_name, method, max_epochs)
                    agent_processes.append(process)
                    agent_conns.append(conn)

                # Start background processes to handle model updates and metrics collection
                model_update_process = Process(
                    target=update_model, args=(agent_conns, method, env_name))
                model_update_process.start()
                print(f"Model update process started for {method}.")

                metrics_process = Process(
                    target=continuous_update, args=(method, env_name, num_instances))
                metrics_process.start()
                print("Metrics aggregation process started.")

            elif command == "stop":
                # Stop all active training processes
                print("Stopping all agents.")
                for process in agent_processes:
                    # Terminate each process
                    process.terminate()
                # Clear the list to start fresh
                agent_processes.clear()

            elif command == "exit":
                # Exit the program stopping all processes
                print("Exiting. Stopping all agents and update processes.")
                for process in agent_processes:
                    # Ensure all agent processes are terminated
                    process.terminate()
                # Clear the process list
                agent_processes.clear()
                # Break the loop to exit the program
                break

    finally:
        # Ensure all processes are terminated on exit to prevent orphan processes
        for process in agent_processes:
            if process.is_alive():
                process.terminate()
        print("All processes have been terminated.")


if __name__ == '__main__':
    # Ensure that all directories exist or create them
    ensure_directories_exist()
    # Start the user interaction loop
    user_interaction()
