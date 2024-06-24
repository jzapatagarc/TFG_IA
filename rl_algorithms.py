import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict
import gymnasium as gym
from agent import ActorCritic, RolloutBuffer
import socket
import pickle

# Nombre alumno: José David Zapata García
# Usuario campus: jzapatagarc


class ExperienceDataset(Dataset):
    """
    A PyTorch Dataset for loading experiences directly from a RolloutBuffer.
    This dataset facilitates easy integration with PyTorch DataLoader for efficient training.

    Args:
        buffer (RolloutBuffer): A buffer that stores training experiences.
    """

    def __init__(self, buffer: RolloutBuffer):
        """
        Initializes the dataset using data from a RolloutBuffer.

        Args:
            buffer (RolloutBuffer): The buffer from which to load experiences.
        """
        self.buffer = buffer
        # Retrieve all data from the buffer.
        self.data = buffer.get_data()

    def __len__(self) -> int:
        """
        Returns the total number of experiences in the buffer.

        Returns:
            int: The number of experiences.
        """
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single experience at a specific index.

        Args:
            idx (int): The index of the experience to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing all components of an experience,
                                     such as state, action, etc.
        """
        # Return a dictionary of tensors, each tensor corresponds to a data point at the index `idx`.
        return {
            'state': self.data['states'][idx],
            'action': self.data['actions'][idx],
            'log_prob': self.data['log_probs'][idx],
            'value': self.data['values'][idx],
            'return': self.data['returns'][idx],
            'advantage': self.data['advantages'][idx]
        }


def save_model(model: nn.Module, method: str, env_name: str, model_path: str):
    """
    Saves the specified model under a unique filename that includes the method and environment name.
    Existing models with the same name are removed to ensure only the latest model is saved.

    Args:
        model (nn.Module): The model to save.
        method (str): The training method, used as part of the filename.
        env_name (str): The environment name, used as part of the filename.
        model_path (str): The path to the directory where the model should be saved.
    """
    # Construct the filename pattern to find existing models to remove
    model_files = [f for f in os.listdir(model_path) if f.startswith(
        method) and f.endswith(env_name + ".pth")]

    # Remove any existing models that match the pattern
    for model_file in model_files:
        os.remove(os.path.join(model_path, model_file))
        print(f"Removed old model {model_file}")

    # Save the new model with a filename that includes the method and environment name
    model_filename = f"{method}_{env_name}.pth"
    torch.save(model.state_dict(), os.path.join(model_path, model_filename))
    print(f"Model saved as {model_filename}")


def apply_a3c_gradients(gradients: Dict[str, torch.Tensor], model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, torch.Tensor]:
    """
    Apply A3C gradients to the global model received from multiple agents.

    Args:
        gradients (Dict[str, torch.Tensor]): Gradients to apply, mapped by parameter names.
        model (nn.Module): The global model to which gradients will be applied.
        optimizer (optim.Optimizer): The optimizer managing the model's parameters.

    Returns:
        Dict[str, torch.Tensor]: The updated parameters of the model after applying gradients.
    """
    # Reset existing gradients in the optimizer to avoid accumulation from previous updates
    optimizer.zero_grad()

    # Apply each received gradient to the corresponding parameter in the model
    missing_gradients = []
    for name, param in model.named_parameters():
        if name in gradients:
            # Ensure the gradient tensor is detached and resides on the same device as the model parameter
            grad = gradients[name].detach()
            grad = grad.to(param.device)
            param.grad = grad
        else:
            # Collect names of parameters for which no gradient was received
            missing_gradients.append(name)

    # Check for any missing gradients and handle accordingly
    if missing_gradients:
        print(
            f"Warning: No gradients received for parameters: {missing_gradients}")

    # Perform a single optimization step to update the model parameters
    optimizer.step()

    return {name: param.data for name, param in model.named_parameters()}


def apply_ppo_experiences(experiences: Dict[str, torch.Tensor], model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, torch.Tensor]:
    """
    Apply PPO experiences to the global model using the Proximal Policy Optimization (PPO) algorithm updates.

    Args:
        experiences (Dict[str, torch.Tensor]): Dictionary containing tensors of states, actions, 
                                               log probabilities, values, returns, and advantages from rollout.
        model (nn.Module): The policy and value network.
        optimizer (optim.Optimizer): Optimizer for the model.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of updated model parameters.
    """
    # Extract components from the experiences dictionary
    states = experiences['states']
    actions = experiences['actions']
    old_log_probs = experiences['log_probs']
    values = experiences['values']
    returns = experiences['returns']
    advantages = experiences['advantages']

    # Reset the optimizer's gradient buffer to avoid accumulation from previous updates
    optimizer.zero_grad()

    # Forward pass to get new action probabilities and state values
    new_action_probs, new_values = model(states)

    # Calculate the log probabilities of the actions taken, ensure actions are in the correct format
    new_log_probs = torch.log(new_action_probs.gather(
        1, actions.unsqueeze(-1).long()).squeeze(-1))

    # Calculate the ratio of new to old probabilities, and apply the clipping technique
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * advantages
    # Minimize the negative of the clipped surrogate objective
    actor_loss = -torch.min(surr1, surr2).mean()

    # Calculate the critic loss using MSE between predicted values and the computed returns
    critic_loss = F.mse_loss(new_values.squeeze(), returns)

    # Entropy bonus to encourage exploration
    # Small epsilon to prevent log(0)
    entropy_bonus = -(new_action_probs *
                      torch.log(new_action_probs + 1e-10)).mean()

    # Total loss combines actor loss, critic loss, and entropy bonus
    loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_bonus

    # Backpropagate the loss, compute gradients
    loss.backward()

    # Update the model parameters
    optimizer.step()

    # Return the updated model parameters
    return {name: param.data for name, param in model.named_parameters()}
