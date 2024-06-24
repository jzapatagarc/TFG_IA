import os
import argparse
import pickle
import gymnasium as gym
import torch
import torch.nn as nn
from torch.optim import RMSprop
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import time
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import uuid
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from typing import Any, Dict, Tuple, Optional, List
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from gymnasium.wrappers import RecordEpisodeStatistics

# Nombre alumno: José David Zapata García
# Usuario campus: jzapatagarc

# Paths configuration for the models and data
MODEL_SAVE_PATH = '/app/local_models'  # Path to save the local model
# Path to load the global model
MODEL_LOAD_PATH = '/home/david/Documents/TFG/global'
DATA_PATH = '/home/david/Documents/TFG/data'
METRICS_PATH = '/home/david/Documents/TFG/metrics'

last_loaded_model = None


class RolloutBuffer:
    """
    RolloutBuffer is a class that stores the experiences of an agent during training.
    This includes observations, actions, rewards, and other relevant data.

    Attributes:
        buffer_size (int): Maximum size of the buffer to store experiences.
        observation_space (gym.Space): The observation space of the environment.
        action_space (gym.Space): The action space of the environment.
        method (str): The training method used.
        gamma (float): The discount factor for future rewards.
        gae_lambda (float): The lambda for Generalized Advantage Estimation (GAE).
        use_gae (bool): Flag to determine whether to use GAE.
        env_name (str): The name of the environment.
        score_threshold (int): The score threshold for success in the environment.
    """

    def __init__(self, buffer_size: int, observation_space: gym.Space, action_space: gym.Space, method: str, gamma: float = 0.9, gae_lambda: float = 0.95, use_gae: bool = True, env_name: str = '', score_threshold: int = 195):
        """Initialize the Rollout Buffer with additional parameters for score tracking."""
        # Creating tensors to store experiences
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

        # Configuration parameters for calculations
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        # Pointer to the current insert position in the buffer
        self.ptr = 0
        # Start index for the next calculation of returns
        self.path_start_idx = 0
        self.max_size = buffer_size
        self.method = method
        self.env_name = env_name

        # Runtime data for metrics
        # Scores per episode
        self.episode_scores = []
        # For LunarLander-v2
        self.successful_landings = 0
        self.total_episodes = 0
        self.score_threshold = score_threshold
        # Track when score threshold is first reached
        self.threshold_reached_times = []
        # Steps in the current episode
        self.current_episode_steps = 0
        # Track duration of each training epoch
        self.epoch_times = []

    def insert(self, state: torch.Tensor, action: int or torch.Tensor, reward: float, done: bool, log_prob: float, value: float):
        """
        Inserts new experience data into the buffer.

        Args:
            state (torch.Tensor): The state observed by the agent.
            action (int or torch.Tensor): The action taken by the agent.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended.
            log_prob (float): The log probability of the action.
            value (float): The value estimate from the critic network.
        """
        if self.ptr < self.max_size:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.log_probs[self.ptr] = log_prob
            self.values[self.ptr] = value
            self.ptr += 1

            # Update score tracking for the current episode
            if len(self.episode_scores) == self.total_episodes:
                self.episode_scores.append(reward)
            else:
                self.episode_scores[-1] += reward
            self.current_episode_steps += 1

            # Check if the score threshold is reached for the first time
            if self.episode_scores[-1] >= self.score_threshold and (len(self.threshold_reached_times) == self.total_episodes):
                self.threshold_reached_times.append(self.current_episode_steps)

            if done:
                self.total_episodes += 1
                self.episode_scores.append(0)
                self.current_episode_steps = 0  # Reset step count for the new episode

        else:
            print("Buffer is full.")

    def start_epoch(self):
        """Records the start time of an epoch for measuring duration."""
        self.start_time = time.time()

    def finish_epoch(self):
        """
        Records the end of an epoch, calculates its duration, and prints it.
        Resets the start time for the next epoch.
        """
        if self.start_time is not None:
            epoch_duration = time.time() - self.start_time
            self.epoch_times.append(epoch_duration)
            print(f"Epoch finished in {epoch_duration:.2f} seconds.")
            self.start_time = None

    def get_metrics(self) -> dict:
        """
        Computes and returns various training metrics based on the data collected.

        Returns:
            dict: A dictionary containing metrics such as average score, success rate,
            and training time.
        """
        training_time = sum(self.epoch_times)

        # Initialize metrics that may not apply to all environments
        average_time_to_threshold = None
        success_rate = None

        # Calculate average time to threshold for CartPole-v1
        if self.env_name == 'CarRacing-v0' and self.threshold_reached_times:
            average_time_to_threshold = sum(
                self.threshold_reached_times) / len(self.threshold_reached_times)

        # Calculate success rate for LunarLander-v2
        if self.env_name == 'LunarLander-v2' and self.total_episodes > 0:
            success_rate = self.successful_landings / self.total_episodes

        metrics = {
            'total_episodes': self.total_episodes,
            'average_time_to_threshold': average_time_to_threshold,
            'success_rate': success_rate,
            'training_time': training_time,
        }

        # Remove None entries to clean up the metrics dictionary
        return {k: v for k, v in metrics.items() if v is not None}

    def send_metrics_to_server(self, metrics_data: dict):
        """
        Serializes and sends accumulated metrics to the server for further analysis.

        Dnamically constructs a filename using the method and environment name, the metrics
        data is then serialized using pickle and written to a file.

        Args:
            metrics_data (dict): The metrics data dictionary containing values that need to be analyzed.
        """
        # Use self.method and self.env_name to dynamically set the file name and method
        env_name = self.env_name
        method_name = self.method
        timestamp = int(time.time())
        # Construct the file name with the method, environment, and timestamp
        file_name = f"{method_name}_{env_name}_metrics_{timestamp}.pkl"
        file_path = os.path.join(METRICS_PATH, file_name)

        # Open the file at file_path in write-binary ('wb') mode.
        with open(file_path, 'wb') as f:
            pickle.dump(metrics_data, f)
        print(f"Metrics data sent to server with file: {file_name}")

    def process_for_a3c(self, gradients: Dict[str, torch.Tensor]):
        """
        Processes the buffer data specifically for the A3C training method after an episode ends.

        This method completes the path computation for the current episode using the finish_path method,
        sends the gradients to the server for global model updates, and resets the buffer pointer to zero.

        Args:
            gradients (Dict[str, torch.Tensor]): Gradients to be sent to the server.
        """
        self.finish_path(last_value=0)
        self.send_gradients_to_server(gradients)
        self.ptr = 0

    def process_for_ppo(self):
        """
        Processes the buffer data for the PPO training method either when the buffer is full or an episode ends.

        Checks if the buffer is at maximum capacity or the last experience recorded was the end of an episode. If true,
        it finalizes the path computations, sends the experience data to the server, and resets the buffer index.
        """
        if self.ptr == self.max_size or self.dones[self.ptr - 1]:
            self.finish_path(last_value=0)
            self.send_experiences_to_server()
            self.ptr = 0

    def send_experiences_to_server(self):
        """
        Serializes and sends experience data to the server using a file.

        This function generates a unique file name using a UUID, combines it with the environment name,
        and dumps the buffer's data into a file. This data is then available for the server to process.
        """
        # Generate a unique identifier for the file
        unique_id = uuid.uuid4()
        data = self.get_data()

        env_name = self.env_name if isinstance(
            self.env_name, str) else self.env_name.spec.id
        file_path = os.path.join(
            DATA_PATH, f"ppo_data_{env_name}_{unique_id}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump({
                'type': 'ppo',
                'model_data': data,
                'env_name': self.env_name if isinstance(self.env_name, str) else self.env_name.spec.id
            }, f)
        print(f"Experiences data sent to server with file: {file_path}")

    def send_gradients_to_server(self, gradients: Dict[str, torch.Tensor]):
        """
        Serializes and sends gradient data to the server using a file.

        Similar to sending experiences, it generates a unique file name with a UUID,
        and stores the gradients data into a file for the server's global model update process.

        Args:
            gradients (Dict[str, torch.Tensor]): Gradients to be sent to the server.
        """
        unique_id = uuid.uuid4()
        env_name = self.env_name if isinstance(
            self.env_name, str) else self.env_name.spec.id
        print(self.env_name)
        print(env_name)
        file_path = os.path.join(
            DATA_PATH, f"a3c_data_{env_name}_{unique_id}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump({
                'type': 'a3c',
                'model_data': gradients,
                'env_name': self.env_name if isinstance(self.env_name, str) else self.env_name.spec.id
            }, f)

    def get_data(self) -> dict:
        """
        Retrieves the buffered data up to the current pointer position.

        Raises:
            ValueError: If the buffer does not contain enough data to fill up to its maximum size.

        Returns:
            dict: A dictionary containing slices of data arrays up to the current buffer pointer.
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

    def discounted_cumsum(self, x: torch.Tensor, discount: float) -> torch.Tensor:
        """
        Calculates the discounted cumulative sum of rewards or advantages.

        Args:
            x (torch.Tensor): Input tensor of rewards or values.
            discount (float): Discount factor to be applied.

        Returns:
            torch.Tensor: Tensor of the same shape as input with discounted cumulative sums.
        """
        result = torch.zeros_like(x)
        addend = 0
        for t in reversed(range(len(x))):
            result[t] = x[t] + discount * addend
            addend = result[t]
        return result

    def finish_path(self, last_value: float = 0):
        """
        Completes the calculations for the current episode path in the buffer,
        computing the returns and advantages using Generalized Advantage Estimation (GAE) if enabled.

        Args:
            last_value (float): The bootstrap value used for calculating returns at the end of the path.
        """
        # Define the slice of the buffer to process.
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

            # Calculate temporal differences
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

    def __len__(self) -> int:
        """
        Returns the current number of entries in the buffer.

        Returns:
            int: The current number of entries stored in the buffer.
        """
        return self.ptr


class ActorCritic(nn.Module):
    """
    An implementation of the Actor-Critic network, a neural network used in reinforcement learning
    that contains both policy (actor) and value (critic) network heads.

    Attributes:
        common (nn.Sequential): Shared layers of the neural network.
        actor (nn.Linear): The actor head that outputs action probabilities.
        critic (nn.Linear): The critic head that outputs a value estimate.
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initializes the ActorCritic network with specified number of inputs and outputs.

        Args:
            num_inputs (int): The number of input features.
            num_outputs (int): The number of possible actions (output dimension of the actor).
        """
        super(ActorCritic, self).__init__()
        # Common layers that both the actor and critic will share.
        self.common = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.LeakyReLU(0.01),   # LeakyReLU to prevent dead neurons

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01)
        )

        # Actor head: outputs the probability distribution over actions
        self.actor = nn.Linear(256, num_outputs)
        # Critic head: outputs a single value estimating the state value
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
            Defines the forward pass of the Actor-Critic model.

            Args:
                x (torch.Tensor): The input tensor containing features of the environment state.

            Returns:
                tuple: A tuple containing:
                    - action_probs (torch.Tensor): The probabilities of each action.
                    - state_values (torch.Tensor): The value estimate of the input state.
            """
        # Pass input through the common layers
        x = self.common(x)

        x = F.relu(x)
        # Softmax to get probabilities from the actor output
        action_probs = F.softmax(self.actor(x), dim=1)
        # Get the value estimate from the critic output
        state_values = self.critic(x)
        return action_probs, state_values


class LightningActorCritic(pl.LightningModule):
    """
    PyTorch Lightning module for training an actor-critic model using methods like PPO and A3C.
    It leverages the capabilities of PyTorch Lightning for efficient training loops, logging, and model management.
    """

    def __init__(self, env: gym.Env, model: torch.nn.Module, memory: RolloutBuffer, hparams: Dict[str, Any], method: str = 'ppo'):
        """
        Initializes the LightningActorCritic module with specified environment, model, memory buffer, and hyperparameters.

        Args:
            env (gym.Env): The environment object from which data is generated.
            model (torch.nn.Module): The actor-critic network model.
            memory (RolloutBuffer): A buffer object for storing rollout data.
            hparams (Dict[str, Any]): Hyperparameters necessary for the model's operation.
            method (str): The training methodology to use. Defaults to 'ppo'.
        """
        super(LightningActorCritic, self).__init__()
        self.env = env
        self.model = model
        self.memory = memory
        # Automatically logs and saves hyperparameters for future use.
        self.save_hyperparameters(hparams)
        self.method = method
        # A list to store the sum of rewards per batch for monitoring.
        self.total_rewards = []
        self.epoch_rewards = 0

        # Assert necessary parameters are included for specific methods.
        if self.method == 'ppo':
            assert 'clip_param' in hparams, "clip_param is required for PPO"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): Input state from the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the action probabilities and state values.
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """
        Performs a single training step.

        Args:
            batch (Tuple[torch.Tensor]): The batch of data to train on.
            batch_idx (int): The index of the batch.

        Returns:
            STEP_OUTPUT: The output after a training step, typically the loss.
        """
        total_reward = 0

        states, actions, rewards, dones, log_probs, values, returns, advantages = batch
        new_action_probs, new_values = self.model(states)

        # Use categorical distribution to calculate the log probabilities of the actions taken.
        dist = torch.distributions.Categorical(new_action_probs)
        new_log_probs = dist.log_prob(actions)

        # Sample an action for each state in the batch
        sampled_actions = dist.sample()

        for action in sampled_actions:
            next_state, reward, terminated, truncated, info = self.env.step(
                action.item())
            total_reward += reward  # Accumulate reward

            if terminated or truncated:
                # Episode has ended
                self.log('episode_total_reward', total_reward,
                         on_step=False, on_epoch=True, logger=True)
                self.env.reset()
                total_reward = 0

        # Accumulate rewards for the epoch
        self.epoch_rewards += rewards.sum().item()

        # Calculate loss based on the method specified.
        if self.method == 'a3c':
            loss = self.calculate_loss_a3c(
                log_probs, new_log_probs, values, returns, advantages)
        elif self.method == 'ppo':
            loss = self.calculate_loss_ppo(
                log_probs, new_log_probs, values, returns, advantages, self.hparams['clip_param'])

        return loss

    def calculate_loss_a3c(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, values: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the A3C method.

        Args:
            old_log_probs (torch.Tensor): Log probabilities of the actions under the old policy.
            new_log_probs (torch.Tensor): Log probabilities of the actions under the current policy.
            values (torch.Tensor): Value estimates from the critic.
            returns (torch.Tensor): Returns calculated from the rewards.
            advantages (torch.Tensor): Advantage estimates.

        Returns:
            torch.Tensor: The calculated loss, combining actor loss, critic loss, and entropy bonus for exploration.
        """
        # Actor loss encourages the agent to take actions that lead to higher advantage.
        actor_loss = -(advantages * new_log_probs).mean()
        # Critic loss measures how well the value function estimates the actual returns.
        critic_loss = F.mse_loss(values.squeeze(), returns)
        # Entropy bonus to encourage exploration
        entropy_bonus = -0.3 * \
            (new_log_probs * torch.exp(new_log_probs)).mean()
        return actor_loss + 0.5 * critic_loss + entropy_bonus

    def calculate_loss_ppo(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, values: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor, clip_param: float) -> torch.Tensor:
        """
        Calculates the loss using the Proximal Policy Optimization (PPO) clipping method.

        Args:
            old_log_probs (torch.Tensor): Log probabilities of the actions under the old policy.
            new_log_probs (torch.Tensor): Log probabilities of the actions under the current policy.
            values (torch.Tensor): Value estimates from the critic.
            returns (torch.Tensor): Returns calculated from the rewards.
            advantages (torch.Tensor): Advantage estimates.
            clip_param (float): The PPO clipping parameter.

        Returns:
            torch.Tensor: The total loss from the actor and critic components.
        """
        # Calculate the ratio of new to old probabilities for determining how much the policy is allowed to change
        ratios = torch.exp(new_log_probs - old_log_probs)
        # Apply clipping to the ratios within specified bounds to prevent large policy updates
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param,
                            1.0 + clip_param) * advantages
        # The actor loss is the minimum of these two surrogateS
        actor_loss = -torch.min(surr1, surr2).mean()
        # The critic loss computes how well the value head predicts the actual returns, using mean squared error
        critic_loss = F.mse_loss(values.squeeze(), returns)
        # Entropy is added to encourage exploration
        entropy_bonus = -0.3 * \
            (new_log_probs * torch.exp(new_log_probs)).mean()
        return actor_loss + 0.5 * critic_loss + entropy_bonus

    def configure_optimizers(self) -> RMSprop:
        """
        Configures the optimizer used for training.

        Returns:
            RMSprop: The RMSprop optimizer with the learning rate specified in hyperparameters.
        """
        return RMSprop(self.model.parameters(), lr=self.hparams['lr'])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Creates the training DataLoader.

        Returns:
            TRAIN_DATALOADERS: A DataLoader containing the training dataset.
        """
        dataset = RolloutBufferDataset(self.memory)
        batch_size = min(self.hparams['batch_size'], len(dataset))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=11)

    def on_train_epoch_start(self):
        """
        Hook that is called at the start of each training epoch.
        """
        self.memory.start_epoch()

    def on_train_epoch_end(self, outputs: Optional[Any] = None):
        """
        Handles tasks at the end of a training epoch, including data processing, sending data to servers,
        logging metrics, and checking for convergence.

        Args:
            outputs (Optional[Any]): Outputs collected from each training step within the epoch.
        """
        print("Epoch ended, processing data...")

        # Log epoch time
        self.memory.finish_epoch()

        # Check if there's data in the memory to process
        if self.memory.ptr > 0:
            data = self.memory.get_data()

            if self.method == 'ppo':
                # Process data for PPO and send data to server
                self.memory.process_for_ppo()
                print("Sending PPO data to server.")

            elif self.method == 'a3c':
                # For A3C compute gradients and send them to the server
                gradients = self.compute_gradients(data)
                self.memory.process_for_a3c(gradients)
                print("Sending A3C gradients to server.")

        # Retrieve and log metrics from the buffer
        metrics = self.memory.get_metrics()
        metrics['reward'] = sum(self.memory.episode_scores) / \
            len(self.memory.episode_scores)
        metrics['threshold'] = False

        # Check convergence based on specific metrics and log the result
        if self.reached_threshold(metrics):
            metrics['threshold'] = True

        # Send accumulated metrics to the server for and analysis
        self.memory.send_metrics_to_server(metrics)

        # Reset buffer for the next training iteration
        self.total_rewards = []
        self.memory.ptr = 0
        self.memory.path_start_idx = 0
        self.epoch_rewards = 0

        # Re-populate memory with initial experiences for the next epoch
        populate_memory(self.env, self.memory, 10000)

    def reached_threshold(self, metrics: Dict[str, Any]) -> bool:
        """
        Determines if the model has reached threshold for the specific environment.

        Args:
            metrics (Dict[str, Any]): Boolean indicating if threshold has been reached in the epoch.

        Returns:
            bool: True if current reward is equal or higher than threshold, otherwise False.
        """
        if self.env.unwrapped.spec.id == "CartPole-v1":
            return metrics.get('reward', 0) >= 195
        else:
            return metrics.get('reward', 0) >= 200

    def compute_gradients(self, data: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Computes the gradients for the actor-critic model using the provided batch data.

        Args:
            data (Dict[str, Any]): A dictionary containing batched data including states, actions,
                                   rewards, dones, returns, advantages, and log probabilities.

        Returns:
            List[torch.Tensor]: A list of gradients for all trainable parameters in the model.
        """
        # Convert data from NumPy arrays or lists to PyTorch tensors.
        states = torch.tensor(data['states'], dtype=torch.float32)
        actions = torch.tensor(data['actions'], dtype=torch.int64)
        rewards = torch.tensor(data['rewards'], dtype=torch.float32)
        dones = torch.tensor(data['dones'], dtype=torch.bool)
        returns = torch.tensor(data['returns'], dtype=torch.float32)
        advantages = torch.tensor(data['advantages'], dtype=torch.float32)
        log_probs = torch.tensor(data['log_probs'], dtype=torch.float32)

        # Forward pass through the model to obtain new action probabilities and state values.
        new_action_probs, values = self.model(states)

        # Compute the probability of the taken actions.
        dist = torch.distributions.Categorical(new_action_probs)
        new_log_probs = dist.log_prob(actions)

        # Calculate loss
        loss = self.calculate_loss_a3c(
            log_probs, new_log_probs, values, returns, advantages)
        # Perform backpropagation to compute gradients.
        loss.backward()

        # Collect gradients.
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                # Detach gradients and prevent further modification.
                gradients.append(param.grad.clone().detach())
            else:
                # If no gradient for a parameter, append a zero tensor of the same shape.
                gradients.append(torch.zeros_like(param))

        return gradients

    def load_model(self, path: str, current_model_name: str, method: str, env: str) -> str:
        """
        Load the model from the specified path if it exists and if the current model name
        does not match the latest model file. This method looks for the latest model file
        that starts with the method name and compares it to the provided
        current model name.

        Args:
            path (str): The file path where model files are stored.
            current_model_name (str): The name of the current model loaded.
            method (str): The method used to initialize the agent.
            env (str): The environment used to initialize the agent to ensure correct dimensions.

        Returns:
            str: The name of the loaded model, or None if no loading occurred.
        """
        # Check if the specified path exists
        if os.path.exists(path):
            # List all files that start with the method_env prefix
            prefix = f"{method}_{env}_"
            model_files = [f for f in os.listdir(path) if f.startswith(prefix)]
            if model_files:
                # Find the latest model file based on modification time
                latest_model = max(
                    model_files, key=lambda f: os.path.getmtime(os.path.join(path, f)))
                # Check if the current model name matches the latest model file
                if current_model_name != latest_model:
                    # Load the model state from the latest model file if different
                    state_dict = torch.load(os.path.join(path, latest_model))

                    # Ensure the keys are correctly formatted
                    new_state_dict = {}
                    for key in state_dict.keys():
                        new_key = key.replace(
                            'model.', '') if key.startswith('model.') else key
                        new_state_dict[new_key] = state_dict[key]

                    self.model.load_state_dict(new_state_dict)
                    print(f"Agent Model loaded from {latest_model}")
                    return latest_model
                else:
                    print("Current model is up to date.")
                    return current_model_name
            else:
                print(
                    "No model found starting with the method used. Starting with a new model.")
                return None
        else:
            print("Model path does not exist. Starting with a new model.")
            return None


class RolloutBufferDataset(Dataset):
    """
    PyTorch Dataset class for accessing and utilizing experiences stored in a RolloutBuffer.

    This dataset allows for easy integration with PyTorch's DataLoader to enable efficient
    batching and processing of experience data collected.

    Attributes:
        memory (RolloutBuffer): An instance of RolloutBuffer containing training data.
    """

    def __init__(self, memory: RolloutBuffer):
        """
        Initializes the dataset with a memory buffer.

        Args:
            memory (RolloutBuffer): The buffer from which to load data.
        """
        self.memory = memory

    def __len__(self) -> int:
        """
        Returns the number of entries in the memory buffer.

        Returns:
            int: The total number of experiences stored in the buffer.
        """
        return len(self.memory)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves an item from the buffer by its index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the state, action, reward, done flag, log probability,
                   value estimate, return, and advantage for the specified index.

        Raises:
            IndexError: If the index is out of range of the dataset's length.
        """
        if idx >= self.__len__():
            raise IndexError("Index out of range")

        # Retrieve data from the buffer
        data = self.memory.get_data()

        # Extract elements for the given index.
        state = data['states'][idx]
        action = data['actions'][idx]
        reward = data['rewards'][idx]
        done = data['dones'][idx]
        log_prob = data['log_probs'][idx]
        value = data['values'][idx]
        return_ = data['returns'][idx]
        advantage = data['advantages'][idx]

        return state, action, reward, done, log_prob, value, return_, advantage


def populate_memory(env: gym.Env, memory: RolloutBuffer, num_initial_experiences: int):
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
        numeric_state = state[0]

    for index in range(num_initial_experiences):
        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take the action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Check if the episode is done
        done = terminated or truncated

        if isinstance(next_state, tuple):
            # Extract the numeric part only
            next_state = next_state[0]

        # Convert numeric state to tensor
        state_tensor = torch.tensor(numeric_state, dtype=torch.float32)
        # Use placeholder values for log_prob and value
        memory.insert(state_tensor, action, reward, done, torch.tensor(
            0.0), torch.tensor(0.0))

        if done:
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
        else:
            # Update numeric_state for next iteration
            numeric_state = next_state

        # Stop populating if the buffer is full
        if memory.ptr >= memory.max_size:
            break


def restart_trainer(trainer: Trainer, max_epochs: int, checkpoint_callback: Callback, logger: CSVLogger) -> Trainer:
    """
    Re-initializes and configures a PyTorch Lightning Trainer with new settings. 
    To reset the training environment after a model update or between training phases.

    Args:
        trainer (Trainer): The existing trainer instance that will be reconfigured.
        max_epochs (int): The maximum number of epochs for the new training session.
        checkpoint_callback (Callback): A callback instance for handling checkpoints during training.
        logger (Logger): A logging interface compatible with PyTorch Lightning.

    Returns:
        Trainer: A new PyTorch Lightning Trainer instance configured with the specified parameters.
    """
    # Reconfigure the existing trainer with new parameters.
    new_trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        precision=32,
        logger=logger
    )

    return new_trainer


def main(env_name: str, method: str, max_epochs: int, conn: int):
    """
    Main training function for initializing and running the training loop.

    Args:
        env_name (str): Name of the gym environment to use.
        method (str): Specifies the training method.
        max_epochs (int): Maximum number of training epochs.
        conn (int): File descriptor for the pipe connection.

    Steps:
        1. Convert the file descriptor to a connection object.
        2. Set up the environment and model.
        3. Configure and initiate training.
        4. Handle dynamic updates based on incoming messages over the connection.
            4.1 For PPO start training, send data to server and wait for response to resume training
            4.2 For A3C start training, send data and continue training while server send a response
    """
    # Convert the file descriptor into a connection object.
    connection = Connection(conn)
    print(f"Environment name: {env_name}, Type: {type(env_name)}")

    # Initialize the environment and model.
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    memory = RolloutBuffer(10000, env.observation_space, env.action_space,
                           method, use_gae=(method == 'ppo'), env_name=env_name, score_threshold=200 if env_name == 'LunarLander-v3' else 195)

    # Populate the memory with initial data.
    populate_memory(env, memory, 10000)

    # Define hyperparameters and create the Lightning module.
    hparams = {'batch_size': 64, 'lr': 0.1,
               'clip_param': 0.3 if method == 'ppo' else None}
    lightning_model = LightningActorCritic(env, model, memory, hparams, method)

    # Setup logger and checkpointing.
    logger = CSVLogger("/logs", name=method)

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='/home/david/Documents/TFG/model_checkpoints/',
        filename=f"{env_name}-{{epoch:02d}}-loss={{train_loss:.2f}}",
        save_top_k=1,
        mode='min'
    )

    # Load latest global model to resume training
    last_loaded_model = lightning_model.load_model(
        MODEL_LOAD_PATH, None, method, env_name)

    # Initialize the trainer with the specified configurations.
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        precision=32,
        logger=logger
    )

    print(f"Memory size after population: {len(memory)}")

    # Start the training loop.
    if method == 'ppo':
        # Training loop for PPO.
        trainer.fit(lightning_model)
        # Listen for commands over the connection.
        while True:
            message = connection.recv()
            # Wait till a new model is sent from the server to resume training
            if message['command'] == 'update':
                trainer = restart_trainer(
                    trainer, max_epochs, checkpoint_callback, logger)
                last_loaded_model = lightning_model.load_model(
                    MODEL_LOAD_PATH, last_loaded_model, method, env_name)
                connection.send({'status': 'params_updated'})
                trainer.fit(lightning_model)
            elif message['command'] == 'stop':
                break
    elif method == 'a3c':
        # Continuous training loop for A3C.
        try:
            while True:
                trainer = restart_trainer(
                    trainer, max_epochs, checkpoint_callback, logger)
                trainer.fit(lightning_model)  # Continuous training.
                if connection.poll():  # Check for new messages.
                    message = connection.recv()
                    # Continue training even if a new model is not sent from the server
                    if message['command'] == 'update':
                        last_loaded_model = lightning_model.load_model(
                            MODEL_LOAD_PATH, last_loaded_model, method, env_name)
                    elif message['command'] == 'stop':
                        break
        finally:
            connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run distributed RL training.")
    parser.add_argument('--env_name', type=str, required=True,
                        help='Name of the gym environment.')
    parser.add_argument('--method', type=str, required=True,
                        choices=['ppo', 'a3c'], help='Training method.')
    parser.add_argument('--max_epochs', type=int, required=True,
                        help='Maximum number of training epochs.')
    parser.add_argument('--conn', type=int, required=True,
                        help='File descriptor for the pipe connection.')

    args = parser.parse_args()
    main(args.env_name, args.method, args.max_epochs, args.conn)
