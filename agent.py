import argparse
import os
import threading
import time
import socket
import pickle
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Global Variables: Hyperparameters for the agent's learning and exploration
BATCH_SIZE = 256  # Number of experiences sampled from memory to learn the policy
GAMMA = 0.75  # Discount factor for future rewards

# Paths configuration for the models
MODEL_SAVE_PATH = '/app/local_models'  # Path to save the local model
MODEL_LOAD_PATH = '/app/global_models'  # Path to load the global model

STOP_EVENT = threading.Event()  # Event to signal when to stop training


def check_for_model_update(model):
    """
    Check the server for an updated model and load it if available.

    This function connects to the server specified in the model's server_address attribute.
    It sends a request to check for an update, receives the updated model state if available,
    and loads the new state into the model.
    """
    # Create a socket object using IPv4 and TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to the server using the address provided in the model
        sock.connect(model.server_address)
        # Send a request to the server to check for an update
        sock.sendall(b"check_update")
        # Receive data from the server (up to 4096 bytes)
        data = sock.recv(4096)
        # If data is received from the server
        if data:
            # Deserialize the received data to get the new model state
            new_state_dict = pickle.loads(data)
            # Load the new state into the model
            model.load_state_dict(new_state_dict)
            # Print a message indicating the model has been updated
            print("Updated model received from server.")


class RolloutBuffer:
    """A buffer for storing trajectory data for PPO or A3C training.

    This buffer stores states, actions, rewards, and other data necessary for training
    Proximal Policy Optimization (PPO) or Asynchronous Advantage Actor-Critic (A3C) models.
    """

    def __init__(self, buffer_size, observation_space, action_space, gamma=0.99, gae_lambda=0.95, use_gae=True, method='ppo'):
        """Initialize the rollout buffer.

        Args:
            buffer_size (int): Maximum size of the buffer.
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation (GAE).
            use_gae (bool): Whether to use GAE for advantage calculation.
            method (str): The training method to use ('ppo' or 'a3c').
        """
        # Initialize buffers for states, actions, rewards, dones, log probabilities, values, returns, and advantages
        self.states = torch.zeros(
            (buffer_size, *observation_space.shape), dtype=torch.float32)
        self.actions = torch.zeros(buffer_size, dtype=torch.int32) if isinstance(
            action_space, gym.spaces.Discrete) else torch.zeros((buffer_size, *action_space.shape), dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)

        # Set discount factor, GAE lambda, and other parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
        self.method = method

    def insert(self, state, action, reward, done, log_prob, value):
        """Insert a new experience into the buffer."""
        if self.ptr < self.max_size:
            # Store the given state, action, reward, done flag, log probability, and value
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.log_probs[self.ptr] = log_prob
            self.values[self.ptr] = value
            self.ptr += 1  # Move the pointer to the next position
            print(f"Data inserted. Current buffer size: {self.ptr}")
        else:
            print("Buffer is full.")

    def process_for_a3c(self):
        """Processes the buffer data for A3C method after an episode ends."""
        # Calculate returns and advantages and send gradients to server
        self.finish_path(last_value=0)
        self.send_gradients_to_server()
        # Reset buffer pointer after processing
        self.ptr = 0

    def process_for_ppo(self):
        """Processes the buffer data for PPO method if buffer is full or episode ends."""
        # Check if buffer is full or last experience was terminal
        if self.ptr == self.max_size or self.dones[self.ptr-1]:
            self.finish_path(last_value=0)
            self.send_experiences_to_server()
            # Reset buffer pointer after processing
            self.ptr = 0

    def send_experiences_to_server(self):
        """Send experiences to the server."""
        # Serialize the buffer data to send to the server
        data = pickle.dumps(self.get_data())
        # Establish a connection with the server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.server_address)
            # Send the serialized data
            sock.sendall(data)
            # Receive and print the response from the server
            response = sock.recv(1024)
        print("Received response from server:", response.decode())

    def discounted_cumsum(self, x, discount):
        """Calculate the discounted cumulative sum of a tensor.

        Args:
            x (torch.Tensor): The input tensor.
            discount (float): The discount factor.

        Returns:
            torch.Tensor: The discounted cumulative sum.
        """
        # Initialize the result tensor with the same size and type as x
        result = torch.zeros_like(x)
        addend = 0
        # Iterate over the elements of x in reverse order to calculate the discounted cumulative sum
        for t in reversed(range(len(x))):
            result[t] = x[t] + discount * addend
            addend = result[t]
        return result

    def finish_path(self, last_value=0):
        """Finish the path and calculate returns and advantages using Generalized Advantage Estimation (GAE) if enabled."""
        # Path slice to calculate values for
        path_slice = slice(self.path_start_idx, self.ptr)
        # If using GAE, compute advantages and returns differently
        if self.use_gae:
            # Append the last value to rewards and values for calculation
            rewards = torch.cat(
                (self.rewards[path_slice], torch.tensor([last_value], dtype=torch.float32)))
            values = torch.cat((self.values[path_slice], torch.tensor(
                [last_value], dtype=torch.float32)))
            dones = torch.cat(
                (self.dones[path_slice], torch.tensor([False], dtype=torch.bool)))
            # Convert boolean tensor to float for operations
            dones = dones.to(torch.float32)

            deltas = rewards[:-1] + self.gamma * \
                values[1:] * (1 - dones[:-1]) - values[:-1]
            self.advantages[path_slice] = self.discounted_cumsum(
                deltas, self.gamma * self.gae_lambda)
            self.returns[path_slice] = self.discounted_cumsum(rewards, self.gamma)[
                :-1]
        else:
            # If not using GAE, simply calculate cumulative returns from rewards
            rewards = self.rewards[self.path_start_idx:self.ptr]
            self.returns[self.path_start_idx:self.ptr] = self.discounted_cumsum(
                rewards, self.gamma)

        # Reset the path start index for the next batch
        self.path_start_idx = self.ptr

    def get_data(self):
        """Get the data from the buffer.

        Returns:
            dict: A dictionary containing states, actions, rewards, dones, log_probs, values, returns, and advantages.
        """
        if self.ptr < self.max_size:
            raise ValueError(
                f"Insufficient data in buffer to retrieve, only {self.ptr}/{self.max_size} entries filled.")

        data = {
            'states': self.states[:self.ptr],
            'actions': self.actions[:self.ptr],
            'rewards': self.rewards[:self.ptr],
            'dones': self.dones[:self.ptr],
            'log_probs': self.log_probs[:self.ptr],
            'values': self.values[:self.ptr],
            'returns': self.returns[:self.ptr],
            'advantages': self.advantages[:self.ptr]
        }
        return data

    def __len__(self):
        """Get the current size of the buffer.

        Returns:
            int: The number of entries currently in the buffer.
        """
        return self.ptr


class ActorCritic(nn.Module):
    """
    A neural network model with an actor and a critic.

    Args:
        num_inputs (int): Number of input features.
        num_outputs (int): Number of output actions.
    """

    def __init__(self, num_inputs, num_outputs, server_address=('host.docker.internal', 0)):
        """
        Initializes the ActorCritic network with specified number of inputs and outputs, and optionally,
        a server address for communication if required.

        Args:
            num_inputs (int): Number of input features.
            num_outputs (int): Number of output actions.
            server_address (tuple): The server address for network communication, default is local Docker internal host.
        """
        super(ActorCritic, self).__init__()
        # Address for connecting to external processes if needed
        self.server_address = server_address

        # Shared layers for both actor and critic
        # These layers process the input data and serve for the actor and the critic.
        self.common = nn.Sequential(
            nn.Linear(num_inputs, 128),  # First layer with ReLU activation
            nn.ReLU(),
            # Second layer to increase representational power
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Actor network for action selection
        # Outputs a vector of action probabilities, based on the state representation
        # from the common layers.
        self.actor = nn.Linear(128, num_outputs)

        # Critic network for state value prediction
        # Iutputs a single value representing the expected return from the current state,
        # based on the common layer outputs.
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass through the network. Takes the input tensor, processes it through shared layers,
        and then separately through the actor and critic to produce action probabilities and state values.

        Args:
            x (Tensor): Input tensor containing state information from the environment.

        Returns:
            Tensor, Tensor: Action probabilities (from the actor) and state value (from the critic).
        """
        x = self.common(x)  # Pass input through common layers
        # Softmax over actor output for probability distribution
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)  # Single value output from critic
        return action_probs, state_values


class LightningActorCritic(pl.LightningModule):
    """
    PyTorch Lightning module for training an actor-critic model using either PPO or A3C methods.
    This class handles the training logic, including the forward pass, loss computation, and optimizer configuration.
    """

    def __init__(self, env, model, memory, hparams, method='ppo'):
        """
        Initializes the LightningActorCritic module. This method sets up the model with its environment,
        learning parameters, and the training method. It also checks if all necessary hyperparameters
        are included for the selected training method.

        Args:
            env: The environment object from which data is generated.
            model: The actor-critic model which will be trained.
            memory: A memory buffer for storing training data.
            hparams: A dictionary containing hyperparameters for training.
            method (str, optional): The training method to use ('ppo' or 'a3c'). Defaults to 'ppo'.
        """
        super(LightningActorCritic, self).__init__()
        self.env = env  # Environment instance from which the agent will learn
        self.model = model  # The model to train
        self.memory = memory  # Rollout buffer to store experience tuples
        # Save hyperparameters for easy access and reproducibility
        self.save_hyperparameters(hparams)
        self.method = method  # Training method, can be 'ppo' or 'a3c'
        # Initialize a list to store total rewards from each training step
        self.total_rewards = []

        if self.method == 'ppo':
            # Ensure PPO-specific parameter is included
            assert 'clip_param' in hparams, "clip_param is required for PPO"

    def forward(self, x):
        """
        Defines the forward pass of the model. This method is called with input data, and it passes
        that data through the model to produce output.

        Args:
            x (Tensor): The input tensor containing state data.

        Returns:
            Tensor: The output from the model, typically action probabilities and state values.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Executes a training step using a batch of data. This method handles the computation of loss based on the
        specified training method and logs the training loss and reward for monitoring.

        Args:
            batch: The batch of data to process. Expected to contain states, actions, rewards, dones, log probabilities,
                   values, returns, and advantages.
            batch_idx (int): The index of the current batch.

        Returns:
            Tensor: The computed loss for the batch, which is used to perform a training update.
        """
        states, actions, rewards, dones, log_probs, values, returns, advantages = batch
        new_action_probs, new_values = self.model(
            states)  # Forward pass through the model

        # Calculate the probability distribution over actions and corresponding new log probabilities
        dist = torch.distributions.Categorical(new_action_probs)
        new_log_probs = dist.log_prob(actions)

        # Compute loss depending on the method specified ('a3c' or 'ppo')
        if self.method == 'a3c':
            loss = self.calculate_loss_a3c(
                log_probs, new_log_probs, values, returns, advantages)
        elif self.method == 'ppo':
            loss = self.calculate_loss_ppo(
                log_probs, new_log_probs, values, returns, advantages, self.hparams['clip_param'])

        # Log the training loss and reward for monitoring purposes
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        total_reward = rewards.sum().item()  # Sum the rewards for the batch
        # Append total reward to the list
        self.total_rewards.append(total_reward)
        self.log('train_reward', total_reward, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss  # Return the computed loss

    def calculate_loss_a3c(self, old_log_probs, new_log_probs, values, returns, advantages):
        """
        Calculate the loss for the Asynchronous Advantage Actor-Critic (A3C) method.
        This method computes separate losses for the actor and the critic, and includes an entropy bonus to encourage exploration.

        Args:
            old_log_probs (Tensor): Log probabilities of actions taken, as recorded by the old policy.
            new_log_probs (Tensor): Log probabilities of actions taken, computed by the new policy.
            values (Tensor): Values estimated by the critic for each state.
            returns (Tensor): Actual returns computed from the environment.
            advantages (Tensor): Advantage estimates used to weight the policy gradient.

        Returns:
            Tensor: The total loss calculated as the sum of actor loss, critic loss, and entropy bonus.
        """
        # Actor loss is calculated to encourage the policy to take actions that lead to higher advantage.
        actor_loss = -(advantages * new_log_probs).mean()
        # Critic loss uses Mean Squared Error to minimize the difference between predicted values and actual returns.
        critic_loss = F.mse_loss(values.squeeze(), returns)
        # Entropy bonus adds a term to the loss to encourage exploration by increasing the entropy of the action distribution.
        entropy_bonus = -0.01 * \
            (new_log_probs * torch.exp(new_log_probs)).mean()
        return actor_loss + 0.5 * critic_loss + entropy_bonus

    def calculate_loss_ppo(self, old_log_probs, new_log_probs, values, returns, advantages, clip_param):
        """
        Calculate the loss for the Proximal Policy Optimization (PPO) method.
        PPO uses a clipped surrogate objective to avoid too large policy updates, which includes an actor loss, a critic loss, and an entropy bonus.

        Args:
            old_log_probs (Tensor): Log probabilities of actions taken, as recorded by the old policy.
            new_log_probs (Tensor): Log probabilities of actions taken, computed by the new policy.
            values (Tensor): Values estimated by the critic for each state.
            returns (Tensor): Actual returns computed from the environment.
            advantages (Tensor): Advantage estimates, which are the differences between returns and value predictions.
            clip_param (float): The clipping parameter 'epsilon', which defines how far away the new policy is allowed to go from the old.

        Returns:
            Tensor: The total loss, which is the sum of the clipped actor loss, the critic loss, and an entropy bonus.
        """
        # Calculate the ratio of new to old probabilities for taken actions, and apply clipping to this ratio.
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param,
                            1.0 + clip_param) * advantages
        # Actor loss is the minimum of unclipped and clipped surrogate objectives, negated to perform gradient ascent.
        actor_loss = -torch.min(surr1, surr2).mean()
        # Critic loss minimizes the squared error between estimated values and returns.
        critic_loss = F.mse_loss(values.squeeze(), returns)
        # Entropy bonus to promote exploration.
        entropy_bonus = -0.01 * \
            (new_log_probs * torch.exp(new_log_probs)).mean()
        return actor_loss + 0.5 * critic_loss + entropy_bonus

    def configure_optimizers(self):
        """
        Setup the optimizer for training the model. This method specifies which optimizer to use and its parameters,
        such as the learning rate.

        Returns:
            torch.optim.Optimizer: The optimizer configured with model parameters and learning rate.
        """
        # Initialize the Adam optimizer with the model parameters and the learning rate specified in hyperparameters
        return optim.Adam(self.model.parameters(), lr=self.hparams['lr'])

    def train_dataloader(self):
        """
        Prepare the DataLoader for training. This method sets up the DataLoader with the dataset containing
        training data, batch size, and other parameters necessary for efficient data loading and training.

        Returns:
            DataLoader: The DataLoader configured to fetch data for training with shuffling and parallel processing enabled.
        """
        # Create a dataset from the memory buffer
        dataset = RolloutBufferDataset(self.memory)
        # Return a DataLoader to handle batching, shuffling, and multi-thread data loading
        return DataLoader(dataset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=11)

    def load_model(self, path):
        """
        Load the model from the specified path if it exists. This method looks for the latest model file
        that starts with the method name ('ppo' or 'a3c') and loads it.

        Args:
            path (str): The file path where model files are stored.

        Returns:
            None: Outputs a status message indicating the result of the load operation.
        """
        # Check if the specified path exists
        if os.path.exists(path):
            # List all files that start with the method name ('ppo' or 'a3c')
            model_files = [f for f in os.listdir(
                path) if f.startswith(self.method)]
            if model_files:
                # Find the latest model file based on modification time
                latest_model = max(
                    model_files, key=lambda f: os.path.getmtime(os.path.join(path, f)))
                # Load the model state from the latest model file
                self.load_state_dict(torch.load(
                    os.path.join(path, latest_model)))
                print(f"Model loaded from {latest_model}")
            else:
                # No model files starting with the method name were found
                print(
                    "No model found starting with the method used. Starting with a new model.")
        else:
            # The specified model path does not exist
            print("Model path does not exist. Starting with a new model.")

    def save_model(self, path):
        """
        Save the model at the specified path. This method ensures that the directory exists,
        and saves the current state of the model under a filename that includes the method name.

        Args:
            path (str): The directory path where the model should be saved.

        Returns:
            None: Outputs a status message indicating where the model was saved.
        """
        # Ensure the directory exists, if not, create it
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the model state under a file name that includes the method name
        torch.save(self.state_dict(), os.path.join(
            path, f"{self.method}_model.pth"))
        print(
            f"Model saved in {os.path.join(path, f'{self.method}_model.pth')}")


class A3CAgent(pl.LightningModule):
    """
    Class for training an Asynchronous Advantage Actor-Critic (A3C) model with asynchronous updates.
    This class manages the entire training process, including environment interaction,
    gradient computation, and asynchronous updates of the model.
    """

    def __init__(self, env_name, lr):
        """
        Initializes the A3C agent with a specific environment and learning rate.

        Args:
            env_name (str): Name of the gym environment to be used for training.
            lr (float): Learning rate for the optimizer.
        """
        super(A3CAgent, self).__init__()
        # Create the environment based on the specified name
        self.env = gym.make(env_name)
        # Initialize the ActorCritic model using the environment's observation and action spaces
        self.model = ActorCritic(
            self.env.observation_space.shape[0], self.env.action_space.n)
        # Initialize a buffer to store experiences
        self.buffer = RolloutBuffer(
            1000, self.env.observation_space, self.env.action_space)
        self.lr = lr  # Set the learning rate

    def forward(self, x):
        """
        Defines the forward pass of the model. This method is called during training and inference
        to get the model's output based on input data.

        Args:
            x (Tensor): Input tensor containing state information from the environment.

        Returns:
            Tensor: The output from the model, action probabilities and state values.
        """
        return self.model(x)

    def compute_gradients(self, batch):
        """
        Computes gradients for a batch of data. This method is essential for updating the model
        parameters based on the loss computed from the batch of experiences.

        Args:
            batch: A batch of data including states, actions, rewards, and dones.

        Returns:
            Tensor: The total loss for the batch, combining actor and critic losses.
        """
        states, actions, rewards, dones = batch
        # Obtain the probabilities and values from the model for the current states
        probs, values = self(states)
        # Use the last state to predict the next values
        _, next_values = self(states[1:] + [states[0]])

        # Compute the returns by adding rewards to the discounted next values
        returns = rewards + self.gamma * next_values * (1 - dones)
        # Calculate advantages as the difference between returns and current estimated values
        advantages = returns - values

        # Compute log probabilities for the taken actions
        log_probs = torch.log(probs.gather(
            1, actions.unsqueeze(-1)).squeeze(-1))
        # Calculate actor loss as negative log probabilities weighted by advantages
        actor_loss = -(log_probs * advantages.detach()).mean()
        # Calculate critic loss as the mean squared error of the advantages
        critic_loss = advantages.pow(2).mean()

        return actor_loss + critic_loss  # Return the combined loss

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step including computing gradients based on the provided batch of data
        and logging the loss.

        Args:
            batch: The batch of data to process. The batch contains environment states, actions, rewards, and dones.
            batch_idx (int): The index of the current batch within the epoch.

        Returns:
            Tensor: The loss computed from the batch, which will be used by the optimizer to update the model parameters.
        """
        # Compute the gradients for the batch
        loss = self.compute_gradients(batch)
        # Log the training loss using PyTorch Lightning's logging
        self.log('train_loss', loss)
        return loss  # Return the loss for backpropagation

    def train_dataloader(self):
        """
        Returns a DataLoader for training the model. Retrieves batches of data from the memory buffer,
        which are passed to the training step.

        Returns:
            DataLoader: A DataLoader with batches of experience data from the buffer.
        """
        # Create a DataLoader that fetches data from the buffer
        return DataLoader(self.buffer, batch_size=32)

    def configure_optimizers(self):
        """
        Configures the optimizer for training the model. This method specifies which optimizer to use
        and its parameters, such as the learning rate.

        Returns:
            torch.optim.Optimizer: The optimizer configured with model parameters and learning rate.
        """
        # Initialize the Adam optimizer with the model parameters and the learning rate specified at initialization
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def on_epoch_end(self):
        """
        Called at the end of each training epoch. Sends gradients to the server to perform asynchronous training.
        """
        print("Epoch ended, sending gradients to server...")
        self.send_gradients_to_server()

    def send_gradients_to_server(self):
        """
        Sends computed gradients to the server and updates model parameters based on server response.
        """
        # Compute gradients from the entire buffer
        gradients = self.compute_gradients(self.buffer[:])
        # Serialize the gradients using pickle to prepare them for sending over the network
        data = pickle.dumps(gradients)
        # Open a socket connection to the server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print(f"Connecting to server at {self.model.server_address}")
            # Connect to the server at the specified address
            sock.connect(self.model.server_address)
            sock.sendall(data)  # Send serialized gradient data to the server
            print("Gradients sent to server.")
            # Receive the updated model parameters from the server
            response = sock.recv(1024)
            # Deserialize the response to get updated parameters
            updated_params = pickle.loads(response)
        print("Received updated parameters from server.")
        # Update the local model parameters with the new parameters received from the server
        self.update_model_parameters(updated_params)

    def update_model_parameters(self, new_params):
        """
        Updates the model parameters with new parameters received from the server.

        Args:
            new_params (List[torch.Tensor]): List of tensors containing the updated parameters.
        """
        # Iterate over each parameter in the model and the corresponding new parameter received
        for param, new_param in zip(self.model.parameters(), new_params):
            # Copy the data from the new parameter to the existing parameter in the model
            param.data.copy_(new_param.data)


class RolloutBufferDataset(Dataset):
    """
    PyTorch Dataset for accessing experiences stored in a RolloutBuffer. Provides an interface for model 
    experiences that can be iterated over during training.

    Args:
        memory (RolloutBuffer): An instance of RolloutBuffer that stores the experiences.
    """

    def __init__(self, memory: RolloutBuffer):
        """
        Initializes the dataset with a memory buffer.

        Args:
            memory (RolloutBuffer): The RolloutBuffer object containing training data.
        """
        self.memory = memory

    def __len__(self):
        """
        Returns the total number of experiences stored in the buffer.

        Returns:
            int: The number of experiences in the buffer.
        """
        return len(self.memory)

    def __getitem__(self, idx):
        """
        Retrieves an individual experience by index. This method allows the dataset to be indexed so that
        it can be used by DataLoader to generate batches of data.

        Args:
            idx (int): The index of the experience to retrieve.

        Returns:
            tuple: A tuple containing state, action, reward, done, log_prob, value, return, and advantage,
                   corresponding to the indexed experience.

        Raises:
            IndexError: If the index is out of the range of the buffer size.
        """
        # Ensure the requested index is within the range of available data
        if idx >= self.__len__():
            raise IndexError("Index out of range")

        # Fetch the data using the get_data method from RolloutBuffer, which returns a dictionary of arrays
        data = self.memory.get_data()

        # Extract and return the data corresponding to the given index
        state = data['states'][idx]
        action = data['actions'][idx]
        reward = data['rewards'][idx]
        done = data['dones'][idx]
        log_prob = data['log_probs'][idx]
        value = data['values'][idx]
        return_ = data['returns'][idx]
        advantage = data['advantages'][idx]

        # Return all these components as a tuple
        return state, action, reward, done, log_prob, value, return_, advantage


def populate_memory(env, memory, num_initial_experiences):
    """
    Populate the memory buffer with initial experiences.

    Args:
        env (gym.Env): Gym environment.
        memory (RolloutBuffer): Buffer to store experiences.
        num_initial_experiences (int): Number of experiences to collect.
    """
    # Reset the environment to get the initial state
    state = env.reset()
    print(f"Initial state from reset: {state}, type: {type(state)}")

    # Process the state to exclude non-numeric data
    if isinstance(state, tuple):
        # Assuming the first element is the numeric state and the second is a dictionary
        numeric_state = state[0]
        print(
            f"Numeric part of the state: {numeric_state}, type: {type(numeric_state)}, shape: {numeric_state.shape}")

    for index in range(num_initial_experiences):
        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take the action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Check if the episode is done
        done = terminated or truncated

        if isinstance(next_state, tuple):
            next_state = next_state[0]  # Extract the numeric part only

        # Convert numeric state to tensor
        state_tensor = torch.tensor(numeric_state, dtype=torch.float32)
        memory.insert(state_tensor, action, reward, done, torch.tensor(
            0.0), torch.tensor(0.0))  # Placeholder values for log_prob and value

        if done:
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Extract the numeric part only
        else:
            numeric_state = next_state  # Update numeric_state for next iteration

        if memory.ptr >= memory.max_size:  # Stop populating if the buffer is full
            break

    print(f"Memory populated: {index+1}/{num_initial_experiences} entries")


def main(env_name, model_path=None, method='ppo', server_ip='host.docker.internal', port=0):
    """
    Main function to configure and run the training loop for an actor-critic model.

    Args:
        env_name (str): The name of the Gym environment to use for training.
        model_path (str, optional): Path to save or load the model during training.
        method (str, optional): The training methodology to use ('ppo' or 'a3c'). Defaults to 'ppo'.
        server_ip (str, optional): IP address of the server for distributed training. Defaults to 'host.docker.internal'.
        port (int, optional): Port number for the server. Defaults to 0.

    Overview:
    1. Set up the server address for distributed training.
    2. Initialize the environment, model, and memory buffer.
    3. Populate the memory buffer with initial experiences.
    4. Define hyperparameters for training.
    5. Initialize the training framework with checkpoints and logging.
    6. Load the model if a path is provided.
    7. Run the training loop until a stop event is triggered.
    8. Save the model upon training completion.
    """
    # Set the server address for communication during distributed training
    global SERVER_ADDRESS
    SERVER_ADDRESS = (server_ip, port)

    # Create the Gym environment with the specified environment name
    env = gym.make(env_name)
    # Initialize the ActorCritic model with the server address for distributed updates
    model = ActorCritic(
        env.observation_space.shape[0], env.action_space.n, (server_ip, port))
    # Create a RolloutBuffer with a size of 10000, configured based on the selected method
    memory = RolloutBuffer(10000, env.observation_space,
                           env.action_space, use_gae=(method == 'ppo'))

    # Populate memory with initial experiences if not fully populated
    if len(memory) < memory.max_size:
        populate_memory(
            env, memory, num_initial_experiences=memory.max_size - len(memory))

    # Define hyperparameters for training
    hparams = {
        'batch_size': 256,  # Size of the batch used in each training step
        'gamma': 0.75,      # Discount factor for future rewards
        'lr': 0.001,        # Learning rate for the optimizer
        'clip_param': 0.2 if method == 'ppo' else None  # Clipping parameter for PPO
    }

    # Initialize the appropriate Lightning module based on the training method
    if method == 'ppo':
        lightning_model = LightningActorCritic(
            env, model, memory, hparams, method)
    elif method == 'a3c':
        lightning_model = A3CAgent(env_name, lr=hparams['lr'])

    # Setup checkpoints and logging
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='./model_checkpoints/',
        filename=f"{env_name}-{{epoch:02d}}-loss={{train_loss:.2f}}",
        save_top_k=1,
        mode='min')
    logger = CSVLogger("logs", name=method)

    # Configure the trainer with specified hardware acceleration and precision
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        precision=16 if torch.cuda.is_available() else 32,
        logger=logger)

    # Load the model if a path is specified
    lightning_model.load_model(model_path)

    # Training loop
    while not STOP_EVENT.is_set():
        trainer.fit(lightning_model)
        # Check and apply updates from the server
        check_for_model_update(model)
        time.sleep(10)  # Sleep for a while before next training cycle

    # Save the model at the specified path upon completion
    lightning_model.save_model(model_path)


if __name__ == '__main__':
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser()

    # Add argument for the environment name (mandatory)
    parser.add_argument("--env_name", type=str,
                        help="Name of the environment to train on.", default="LunarLander-v2")

    # Add argument for the model path (optional)
    parser.add_argument("--model_path", type=str,
                        help="Path to load the model checkpoint.", default=None)

    # Add argument for the training method (optional, with default)
    parser.add_argument("--method", type=str, choices=['a3c', 'ppo'], help="Training method to use: 'a3c' or 'ppo'.",
                        default='ppo')

    # Add argument for the server port (optional, with default)
    parser.add_argument("--port", type=int,
                        help="Port number for the server to listen on.", default=0)

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.env_name, args.model_path, args.method)
