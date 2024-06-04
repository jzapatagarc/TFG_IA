import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict
import gym
from agent import ActorCritic, RolloutBuffer
import socket
import pickle


class ExperienceDataset(Dataset):
    """
    PyTorch Dataset for loading experiences from RolloutBuffer.

    Args:
        buffer (RolloutBuffer): Buffer storing experiences.
    """

    def __init__(self, buffer: RolloutBuffer):
        self.buffer = buffer
        self.data = buffer.get_data()

    def __len__(self) -> int:
        """Return the number of experiences in the dataset."""
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an experience at a specific index."""
        return {
            'state': self.data['states'][idx],
            'action': self.data['actions'][idx],
            'log_prob': self.data['log_probs'][idx],
            'value': self.data['values'][idx],
            'return': self.data['returns'][idx],
            'advantage': self.data['advantages'][idx]
        }


def save_model(model, method, env_name, model_path):
    """
    Save the model with a name that starts with the method and ends with the environment name,
    and remove any previous models with the same naming pattern.

    Args:
        model (nn.Module): The model to save.
        method (str): The method name ('a3c' or 'ppo').
        env_name (str): The environment name.
        model_path (str): The directory path where the model will be saved.
    """
    # Find existing model files that match the naming pattern
    model_files = [f for f in os.listdir(model_path) if f.startswith(
        method) and f.endswith(env_name + ".pth")]

    # Remove existing model files
    for model_file in model_files:
        os.remove(os.path.join(model_path, model_file))

    # Save the new model
    model_filename = f"{method}_{env_name}.pth"
    torch.save(model.state_dict(), os.path.join(model_path, model_filename))
    print(f"Model saved as {model_filename}")


def apply_a3c_gradients(gradients: Dict[str, torch.Tensor], model: nn.Module, optimizer: optim.Optimizer):
    """
    Apply A3C gradients to the global model.

    Args:
        gradients (Dict[str, torch.Tensor]): Gradientes to apply.
        model (nn.Module): Global model.
        optimizer (optim.Optimizer): Model's optimizer.

    Returns:
        Dict[str, torch.Tensor]: Parámetros del modelo actualizado.
    """
    # Reset gradients in the optimizer before applying new ones
    optimizer.zero_grad()

    # Iterate over all named parameters in the model
    for name, param in model.named_parameters():
        if name in gradients:
            # Replace the current gradient for each parameter with the new gradient received
            param.grad = gradients[name]

    # Perform a single optimization step to update the model parameters
    optimizer.step()

    # Collect and return the updated parameters
    return {name: param.data for name, param in model.named_parameters()}


def apply_ppo_experiences(experiences: Dict[str, torch.Tensor], model: nn.Module, optimizer: optim.Optimizer):
    """
    Apply PPO experiences to the global model.

    Args:
        experiences (Dict[str, torch.Tensor]): Experiences to apply.
        model (nn.Module): Global model.
        optimizer (optim.Optimizer): Optimizer for the model.

    Returns:
        Dict[str, torch.Tensor]: Updated model parameters.
    """
    # Extract components from the experiences dictionary
    states = experiences['states']
    actions = experiences['actions']
    log_probs = experiences['log_probs']
    values = experiences['values']
    returns = experiences['returns']
    advantages = experiences['advantages']

    # Reset the optimizer's gradient buffer before calculating new gradients
    optimizer.zero_grad()

    # Forward pass through the model to get action probabilities and state value estimates for the given states
    new_action_probs, new_values = model(states)

    # Calculate the new log probabilities for the actions taken
    new_log_probs = torch.log(new_action_probs.gather(
        1, actions.unsqueeze(-1)).squeeze(-1))

    # Calculate the ratio of new to old probabilities, clip it, and calculate the PPO objective
    ratios = torch.exp(new_log_probs - log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # Calculate the critic loss using mean squared error between estimated and actual returns
    critic_loss = F.mse_loss(new_values.squeeze(), returns)

    # Calculate the entropy bonus for encouraging exploration
    entropy_bonus = (new_action_probs * torch.log(new_action_probs)).mean()

    # Combine the actor loss, critic loss, and entropy bonus into total loss
    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

    # Perform backpropagation to compute gradients
    loss.backward()

    # Update model parameters using the optimizer
    optimizer.step()

    # Return the updated model parameters
    return {name: param.data for name, param in model.named_parameters()}


def receive_gradients(port=0) -> Dict[str, torch.Tensor]:
    """
    Receive gradients from the server.

    Args:
        port (int): The port number on which the server listens for incoming gradient updates.

    Returns:
        Dict[str, torch.Tensor]: Gradients received from the server.
    """
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect the socket to the server address and port
        sock.connect(('127.0.0.1', port))
        # Try to receive data from the socket up to 4096 bytes of data.
        data = sock.recv(4096)

    # Deserialize the data received from the server
    gradients = pickle.loads(data)

    # Return the deserialized gradients
    return gradients


def receive_experiences(port=0) -> Dict[str, torch.Tensor]:
    """
    Receive experiences from the server.

    Args:
        port (int): The port number on which the server listens for incoming experiences.

    Returns:
        Dict[str, torch.Tensor]: Experiences received from the server.
    """
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect the socket to the server address and port
        sock.connect(('127.0.0.1', port))
        # Try to receive data from the socket up to 4096 bytes of data.
        data = sock.recv(4096)

    # Return the deserialized experiences
    return pickle.loads(data)


def a3c_training(env_name: str, max_epochs: int, output_checkpoint_path: str):
    """
    Train a model using the A3C algorithm.

    Args:
        env_name (str): Name of the environment to train on.
        max_epochs (int): Number of epochs to train.
        output_checkpoint_path (str): Path to save the trained model.
    """
    # Initialize the environment
    env = gym.make(env_name)
    # Get the number of input features from the environment
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n  # Get the number of actions from the environment

    # Create the A3C model
    model = ActorCritic(input_dim, output_dim)
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set the model to training mode
    model.train()

    # Training loop
    for epoch in range(max_epochs):
        # Receive gradients
        gradients = receive_gradients()
        # Apply received gradients to the model
        updated_params = apply_a3c_gradients(gradients, model, optimizer)

        # Print the progress after each epoch
        print(
            f"Epoch {epoch+1}/{max_epochs}, Updated Parameters: {updated_params}")

    # Save the trained model to the specified path
    save_model(model, 'a3c', env_name, output_checkpoint_path)
    print(f"A3C model saved to {output_checkpoint_path}")


def ppo_training(env_name: str, max_epochs: int, output_checkpoint_path: str):
    """
    Train a model using the PPO algorithm.

    Args:
        env_name (str): Name of the environment to train on.
        max_epochs (int): Number of epochs to train.
        output_checkpoint_path (str): Path to save the trained model.
    """
    # Initialize the training environment
    env = gym.make(env_name)
    # Number of input features from the environment
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n  # Number of possible actions in the environment

    # Initialize the PPO model
    model = ActorCritic(input_dim, output_dim)
    # Set up the optimizer for the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set the model to training mode
    model.train()

    # Loop through the specified number of training epochs
    for epoch in range(max_epochs):
        # Receive training experiences
        experiences = receive_experiences()
        # Apply these experiences to the model
        updated_params = apply_ppo_experiences(experiences, model, optimizer)

        # Print the progress after each epoch
        print(
            f"Epoch {epoch+1}/{max_epochs}, Updated Parameters: {updated_params}")

    # After training, save the model to the specified path
    save_model(model, 'ppo', env_name, output_checkpoint_path)
    print(f"PPO model saved to {output_checkpoint_path}")
