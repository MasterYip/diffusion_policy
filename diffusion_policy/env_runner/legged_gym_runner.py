'''
Author: Raymon Yip 2205929492@qq.com
Date: 2025-03-29 19:34:29
Description: file content
FilePath: /PredictiveDiffusionPlanner_Dev/diffusion_policy/diffusion_policy/env_runner/legged_gym_runner.py
LastEditTime: 2025-03-31 13:10:45
LastEditors: Raymon Yip
'''

import os
import time
import numpy as np
import isaacgym
import torch
import tqdm
from typing import Dict, Optional, List, Tuple

from diffusion_policy.env.legged_gym.legged_gym_env import LeggedGymEnv
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

# Note: The actual import should point to where LeggedGymEnv is located in your project
# This is a placeholder and should be updated when integrating with your project
# from legged_gym_cmp.legged_gym.legged_gym.legged_gym_env import LeggedGymEnv


class LeggedGymRunner(BaseLowdimRunner):
    """
    Runner for legged gym environments that interfaces with diffusion policies.
    """

    def __init__(self,
                 output_dir: str,
                 task_name: str = "elspider_air_flat",
                 n_train: int = 10,
                 n_train_vis: int = 3,
                 train_start_seed: int = 0,
                 n_test: int = 22,
                 n_test_vis: int = 6,
                 test_start_seed: int = 10000,
                 max_steps: int = 1000,
                 n_obs_steps: int = 8,
                 n_action_steps: int = 8,
                 n_latency_steps: int = 0,
                 fps: int = 50,
                 tqdm_interval_sec: float = 5.0,
                 n_envs: int = 4,
                 headless: bool = True,
                 device: Optional[str] = None,
                 ):
        """
        Initialize the LeggedGymRunner.

        Args:
            output_dir: Directory to save outputs
            task_name: Name of the legged gym task
            n_train: Number of training episodes
            n_train_vis: Number of training episodes to visualize
            train_start_seed: Starting seed for training episodes
            n_test: Number of test episodes
            n_test_vis: Number of test episodes to visualize
            test_start_seed: Starting seed for test episodes
            max_steps: Maximum number of steps per episode
            n_obs_steps: Number of observation steps for the policy
            n_action_steps: Number of action steps to predict
            n_latency_steps: Number of latency steps
            fps: Frames per second for visualization
            tqdm_interval_sec: Interval for tqdm updates
            n_envs: Number of parallel environments
            headless: If True, disable visualization
            device: Device to run on (if None, use policy device)
        """
        super().__init__(output_dir)

        self.task_name = task_name
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.fps = fps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device
        self.n_envs = n_envs
        self.headless = headless

        # Environment will be created when run is called to avoid
        # creating it for validation/testing during training
        self.env = None

        # Ensure the observation history is sufficient for the policy
        self.history_len = n_obs_steps

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        """
        Run the policy in the environment.

        Args:
            policy: The policy to run

        Returns:
            Dictionary containing run results
        """
        # Create the environment if it doesn't exist
        if self.env is None:
            self.env = LeggedGymEnv(
                task_name=self.task_name,
                num_envs=self.n_envs,
                headless=self.headless
            )

        # Use policy device if no device specified
        if self.device is None:
            device = policy.device
        else:
            device = torch.device(self.device)

        dtype = policy.dtype

        # Reset the environment
        obs, _ = self.env.reset()

        # Initialize progress bar
        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=f"Evaluating {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec
        )

        # Initialize observation and action history - matching cyber_runner structure
        history = self.n_obs_steps
        
        # Use get_diffusion_observation if available, otherwise use regular obs
        if hasattr(self.env, 'get_diffusion_observation'):
            diffusion_obs = self.env.get_diffusion_observation().to(device)
            obs_dim = diffusion_obs.shape[-1]
        else:
            diffusion_obs = obs.to(device)
            obs_dim = obs.shape[-1]
        
        state_history = torch.zeros((self.env.num_envs, history+1, obs_dim), dtype=dtype, device=device)
        action_history = torch.zeros((self.env.num_envs, history, self.env.num_actions), dtype=dtype, device=device)

        # Initialize state history with current observation - matching cyber_runner pattern
        state_history[:, :, :] = diffusion_obs[:, None, :]

        # Initialize metrics
        episode_lengths = []
        episode_rewards = []
        current_episode_rewards = torch.zeros(self.env.num_envs, device=device)

        step_count = 0
        while step_count < self.max_steps:
            # Prepare observations for policy - USE DELAYED INPUTS like cyber_runner
            obs_dict = {"obs": state_history[:, -policy.n_obs_steps-1:-1, :]}

            # Get actions from policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
                pred_action = action_dict["action_pred"]
                
                # USE THE NEXT PREDICTED ACTION - RHC Framework like cyber_runner
                action = pred_action[:, history:history+1, :]

            # Step environment multiple times if n_action_steps > 1 (matching cyber_runner)
            self.n_action_steps = action.shape[1]
            
            for i in range(self.n_action_steps):
                action_step = action[:, i, :]
                next_obs, rewards, dones, info = self.env.step(action_step)
                
                # Update state and action history - matching cyber_runner roll pattern
                state_history = torch.roll(state_history, shifts=-1, dims=1)
                action_history = torch.roll(action_history, shifts=-1, dims=1)
                
                # Update with diffusion observation if available
                if hasattr(self.env, 'get_diffusion_observation'):
                    state_history[:, -1, :] = self.env.get_diffusion_observation().to(device)
                else:
                    state_history[:, -1, :] = next_obs.to(device)
                
                step_count += 1
                
                # Accumulate rewards
                current_episode_rewards += rewards

                # Handle episode terminations - matching cyber_runner reset logic
                env_ids = torch.nonzero(dones, as_tuple=False).squeeze(1).int()
                if len(env_ids) > 0:
                    # Reset state and action history for terminated episodes
                    current_obs = state_history[:, -1, :].to(device)
                    state_history[env_ids, :, :] = current_obs[env_ids][:, None, :]
                    action_history[env_ids, :, :] = 0.0
                    
                    # Record metrics for terminated episodes
                    for idx in env_ids:
                        episode_rewards.append(current_episode_rewards[idx].item())
                        episode_lengths.append(step_count)
                        print(f"Episode finished with reward {episode_rewards[-1]:.2f} after {episode_lengths[-1]} steps")
                        
                        # Reset rewards for terminated episodes
                        current_episode_rewards[idx] = 0

            # Update progress bar by number of action steps taken
            pbar.update(action.shape[1])

        pbar.close()

        # If any episodes didn't terminate, add their stats too
        for i in range(self.env.num_envs):
            if current_episode_rewards[i] > 0:  # Episode didn't terminate
                episode_rewards.append(current_episode_rewards[i].item())
                episode_lengths.append(self.max_steps)

        # Aggregate metrics
        results = {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
        }

        return results
