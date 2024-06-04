import docker
import time
import os
import glob
from typing import List

# Initialize the Docker client
client = docker.from_env()


def start_docker_instances_with_args(num_instances, image_name, method, env_name, port):
    """
    Start the specified number of Docker containers with the given image and method.

    Args:
        num_instances (int): Number of Docker containers to start.
        image_name (str): Name of the Docker image to use.
        method (str): Method to be used (e.g., 'a3c', 'ppo').
        env_name (str): Name of the environment.
        port (int): Port number on which the server in the container will listen.

    Returns:
        list: List of Docker container instances.
    """
    containers = []  # List to store references to the started containers

    for _ in range(num_instances):
        # Run a Docker container in detached mode with the specified image and configurations
        container = client.containers.run(
            image=image_name,
            detach=True,
            environment={
                'METHOD': method,  # Pass the method as an environment variable
                'ENV_NAME': env_name,  # Pass the environment name as an environment variable
                # Pass the port number as an environment variable
                'PORT': str(port)
            },
            volumes={
                os.path.abspath('local_models'): {
                    'bind': '/app/local_models',  # Bind local models directory to the container
                    'mode': 'rw'  # Allow read/write access
                },
                os.path.abspath('global_models'): {
                    'bind': '/app/global_models',  # Bind global models directory to the container
                    'mode': 'rw'  # Allow read/write access
                }
            },
            network_mode='host',  # Use the host's network stack
            # Map host.docker.internal to the host IP
            extra_hosts={'host.docker.internal': '172.0.0.1'},
            runtime='nvidia',  # Specify the runtime for GPU usage
            device_requests=[docker.types.DeviceRequest(
                count=-1,  # Request all available GPUs
                capabilities=[['gpu']]
            )]
        )
        containers.append(container)  # Add the started container to the list
        # Wait a second before starting the next container to prevent race conditions
        time.sleep(1)

    return containers  # Return the list of started Docker container instances


def stop_docker_instances(containers: List[docker.models.containers.Container]):
    """
    Stop and remove specified Docker containers.

    Args:
        containers (List[docker.models.containers.Container]): List of containers to stop and remove.
    """
    for container in containers:
        # Stop the container
        container.stop()
        # Remove the container
        container.remove()


def check_container_status(container: docker.models.containers.Container) -> str:
    """
    Return the current status of a Docker container.

    Args:
        container (docker.models.containers.Container): Container to check status of.

    Returns:
        str: Status of the container.
    """
    # Reload the container status
    container.reload()
    return container.status


def get_container_logs(container: docker.models.containers.Container) -> str:
    """
    Retrieve logs from a specified Docker container.

    Args:
        container (docker.models.containers.Container): Container to get logs from.

    Returns:
        str: Logs of the container.
    """
    try:
        # Get and decode the logs of the container
        return container.logs().decode('utf-8')
    except docker.errors.NotFound:
        return "Container not found."


def check_for_new_model(directory: str = "./model") -> str:
    """
    Check for the newest model in the specified directory.

    Args:
        directory (str): Directory to check for new models.

    Returns:
        str: Path to the newest model if available, else None.
    """
    # Find all model files in the directory
    model_files = glob.glob(os.path.join(directory, "*.ckpt"))
    if not model_files:
        return None

    # Find the most recently created model file
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model


def distribute_model(containers: List[docker.models.containers.Container], model_path: str):
    """
    Distribute a model file to specified Docker containers.

    Args:
        containers (List[docker.models.containers.Container]): List of containers to distribute the model to.
        model_path (str): Path to the model file to distribute.
    """
    for container in containers:
        with open(model_path, 'rb') as file:
            # Put the model file into the container
            container.put_archive('path', file.read())


def scale_containers(base_image: str, desired_count: int):
    """
    Dynamically scale the number of Docker containers based on desired count.

    Args:
        base_image (str): Base image to use for scaling.
        desired_count (int): Desired number of containers.
    """
    # Get the list of current containers using the base image
    current_containers = client.containers.list(
        filters={"ancestor": base_image})
    current_count = len(current_containers)
    if current_count < desired_count:
        # Start additional containers if needed
        start_docker_instances_with_args(
            desired_count - current_count, base_image)
    elif current_count > desired_count:
        # Stop excess containers if needed
        containers_to_remove = current_containers[desired_count:]
        stop_docker_instances(containers_to_remove)


def cleanup_containers():
    """
    Remove all stopped or exited Docker containers to free up resources.
    """
    for container in client.containers.list(all=True):
        if container.status in ['exited', 'dead']:
            # Remove the container if it is stopped or dead
            container.remove()
