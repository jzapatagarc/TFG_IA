from flask import Flask, request, jsonify
import docker_management as docker_mgmt
import rl_algorithms as rl
import os
import socket
import pickle
from threading import Thread
import torch
import torch.optim as optim
import gym

# Paths configuration for the models
# Path to load the local model
LOCAL_MODELS_PATH = '/home/david/Documents/app/local_models'
# Path to save the global model
GLOBAL_MODEL_PATH = '/home/david/Documents/app/global_models'

# Initialize the Flask application
app = Flask(__name__)


def load_global_model(method, env_name):
    """
    Load the global model based on the specified method and environment, and initialize the optimizer.

    Args:
        method (str): The training method to use, which determines the type of model architecture.
        env_name (str): The name of the gym environment to determine the input and output dimensions for the model.
    """
    global global_model, optimizer  # Declare global variables for the model and optimizer

    # Create a gym environment to obtain input and output dimensions based on the environment's configuration
    env = gym.make(env_name)
    # Input dimensions from the environment's observation space
    input_dim = env.observation_space.shape[0]
    # Output dimensions from the environment's action space
    output_dim = env.action_space.n

    # Initialize the global model using the input and output dimensions
    global_model = rl.ActorCritic(input_dim, output_dim)

    # Path where global models are stored
    model_files = [f for f in os.listdir(
        GLOBAL_MODEL_PATH) if f.startswith(method)]
    # Check if there are any saved models that start with the specified method
    if model_files:
        # If there are saved models, load the most recent model
        latest_model_file = max(model_files, key=lambda f: os.path.getmtime(
            os.path.join(GLOBAL_MODEL_PATH, f)))
        global_model.load_state_dict(torch.load(
            os.path.join(GLOBAL_MODEL_PATH, latest_model_file)))
        print(f"Global model loaded from {latest_model_file}")
    else:
        # If no models are found, start with a new model
        print("No global model found. Starting with a new model.")

    # Initialize the optimizer for the global model
    # Set learning rate for the optimizer
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)


def handle_a3c_updates(port=0):
    """
    Handles incoming A3C gradient updates over a network socket, updates the global model,
    and broadcasts the updated parameters to all connected A3C agents.

    Args:
        port (int): The port number on which the server listens for incoming gradient updates. Defaults to 0
    """
    # Create a socket object using IPv4 and TCP protocol
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to localhost on the specified port
    s.bind(('127.0.0.1', port))
    # Start listening for incoming connections
    s.listen()
    # Retrieve and print the port number
    port = s.getsockname()[1]
    print(f"A3C listening on port: {port}")

    while True:  # Server runs continuously
        # Accept a connection from an A3C agent
        conn, addr = s.accept()
        print(f"A3C connection accepted from: {addr}")
        # Receive data from the connection, expecting a pickle object
        data = conn.recv(4096)
        print("A3C gradients received.")
        # Deserialize the data to get gradients
        gradients = pickle.loads(data)

        # Load the current global model from file
        global_model = torch.load(GLOBAL_MODEL_PATH)
        # Initialize the optimizer with the current model parameters and specified learning rate
        optimizer = optim.Adam(global_model.parameters(), lr=0.001)
        # Apply the received gradients to the global model and get updated parameters
        updated_params = rl.apply_a3c_gradients(
            gradients, global_model, optimizer)
        # Save the updated global model back to the file system
        torch.save(global_model.state_dict(), GLOBAL_MODEL_PATH)
        print("A3C global model updated and saved.")

        # Send the updated model parameters back to the agent
        conn.send(pickle.dumps(updated_params))
        print("A3C updated parameters sent back.")
        # Close the connection
        conn.close()


def handle_ppo_updates(port=0):
    """
    Handles incoming PPO experiences over a network socket, updates the global model,
    and broadcasts the updated parameters to all connected PPO agents.

    Args:
        port (int): The port number on which the server listens for incoming experiences. Defaults to 0
    """
    # Create a TCP/IP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to localhost on the given port
    s.bind(('127.0.0.1', port))
    s.listen()  # Start listening for incoming connections
    # Report the port it's listening on
    print(f"PPO listening on port: {s.getsockname()[1]}")

    while True:
        conn, addr = s.accept()  # Accept a new connection
        print(f"PPO connection accepted from: {addr}")

        data = conn.recv(4096)  # Receive data from the connection
        print("PPO experiences received.")
        experiences = pickle.loads(data)  # Deserialize the received data

        # Load the global model and create an optimizer
        model = torch.load(GLOBAL_MODEL_PATH)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Apply the received experiences to update the model
        updated_params = rl.apply_ppo_experiences(
            experiences, model, optimizer)
        # Save the updated model parameters
        torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
        print("PPO global model updated and saved.")

        # Serialize the updated model parameters and send them back to the agent
        conn.send(pickle.dumps(updated_params))
        print("PPO updated parameters sent back.")
        conn.close()  # Close the connection to handle the next one


@app.route('/start', methods=['POST'])
def start_training():
    """
    Start training environments based on specified parameters, setting up Docker containers.
    This endpoint initializes the global model, starts training sessions according to the specified method (A3C or PPO),
    and manages Docker containers for distributed training.

    Request JSON structure:
        {
            "num_instances": int,    # Number of training instances (containers) to start
            "image_name": str,       # Docker image name to use for the containers
            "method": str,           # Training method, either 'a3c' or 'ppo'
            "env_name": str          # Gym environment name for the training
        }

    Returns:
        JSON: A response with a message indicating success and a list of container IDs if applicable.
    """
    # Parse the JSON data sent with the request
    data = request.json
    method = data['method']
    env_name = data['env_name']

    # Load the global model with the specified method and environment name
    load_global_model(method, env_name)

    # Function that starts Docker containers and returns their instances
    containers = docker_mgmt.start_docker_instances_with_args(
        num_instances=data['num_instances'],
        image_name=data['image_name'],
        method=data['method'],
        env_name=data['env_name'],
        port=data['port']
    )

    # Depending on the method, start the update handling thread
    if method == 'a3c':
        a3c_thread = Thread(target=handle_a3c_updates)
        a3c_thread.start()  # Start thread to handle asynchronous updates from A3C agents
    elif method == 'ppo':
        ppo_thread = Thread(target=handle_ppo_updates)
        ppo_thread.start()  # Start thread to handle updates from PPO agents

    # Gather container IDs to return in the response
    container_ids = [c.id for c in containers]

    # Return a JSON response indicating the training has started and providing the IDs of the containers
    return jsonify({'message': 'Training environments initiated', 'container_ids': container_ids}), 200


@app.route('/stop', methods=['POST'])
def stop_training():
    """
    Stops specified Docker containers based on container IDs provided in the POST request.

    Request JSON structure:
        {
            "containers": List[str]  # List of Docker container IDs to stop
        }

    Returns:
        JSON: A response with a message confirming that the specified containers have been stopped.
    """
    # Extract the list of container IDs from the request JSON body
    container_ids = request.json.get('containers', [])

    # Check if the container list is empty and return an error message if it is
    if not container_ids:
        return jsonify({'error': 'No container IDs provided'}), 400

    # Stop the specified containers
    docker_mgmt.stop_docker_instances(container_ids)

    print(f"Stopped Docker containers with IDs: {container_ids}")

    # Return a JSON response indicating that the containers have been stopped
    return jsonify({'message': 'Containers stopped successfully'}), 200


@app.route('/cleanup', methods=['GET'])
def cleanup():
    """
    Cleans up all stopped or exited Docker containers to free up system resources.

    It starts a cleanup process that removes any Docker containers that have stopped running or have exited.

    Returns:
        A JSON response with a message confirming that the cleanup has been completed.
    """

    # Call the cleanup function to remove stopped or exited containers
    docker_mgmt.cleanup_containers()

    print("Cleanup process initiated for stopped and exited Docker containers.")

    # Return a JSON response indicating that the cleanup has been successfully completed
    return jsonify({'message': 'Cleaned up exited containers successfully'}), 200


if __name__ == '__main__':
    # Starting background threads for handling A3C and PPO updates
    a3c_thread = Thread(target=handle_a3c_updates, args=(0,))
    a3c_thread.daemon = True  # Set as a daemon so it does not block the main thread
    a3c_thread.start()

    ppo_thread = Thread(target=handle_ppo_updates, args=(0,))
    ppo_thread.daemon = True
    ppo_thread.start()

    # Run the Flask application on localhost at an automatically assigned port
    app.run(debug=True, host='127.0.0.1', port=0)
