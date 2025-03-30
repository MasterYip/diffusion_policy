'''
Author: Raymon Yip 2205929492@qq.com
Date: 2025-03-29 19:20:16
Description: file content
FilePath: /PredictiveDiffusionPlanner_Dev/diffusion_policy/diffusion_policy/env/legged_gym/legged_gym_env.py
LastEditTime: 2025-03-30 09:44:01
LastEditors: Raymon Yip
'''
# autopep8: off
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

from typing import Dict, Tuple, Optional, Any, List, Union
import numpy as np
import torch
import time
# autopep8: on


class LeggedGymEnv:
    """A wrapper class for legged gym environments to provide a standardized interface."""

    def __init__(
        self,
        task_name: str = "anymal_c_flat",
        num_envs: Optional[int] = None,
        headless: bool = True,
        sim_device: str = "cuda:0",
        rl_device: str = "cuda:0",
        physics_engine: int = gymapi.SIM_PHYSX,
        seed: Optional[int] = None,
        args=None,
        **kwargs
    ):
        """Initialize the legged gym environment.

        Args:
            task_name: Name of the registered legged gym task
            num_envs: Number of parallel environments to create
            headless: If True, disable visualization
            sim_device: Device to run simulation on ("cuda:0", "cuda:1", "cpu", etc)
            rl_device: Device to run RL algorithm on
            physics_engine: Physics engine to use ("physx" or "flex")
            seed: Random seed
            args: Command line arguments (if None, default args will be used)
        """
        # Create custom args if not provided
        if args is None:
            args = get_args()
            # Override args with provided params
            args.task = task_name
            args.headless = headless
            args.sim_device_type = sim_device.split(':')[0]
            if ':' in sim_device:
                args.compute_device_id = int(sim_device.split(':')[1])
            args.rl_device = rl_device
            args.physics_engine = physics_engine

            if num_envs is not None:
                args.num_envs = num_envs
            if seed is not None:
                args.seed = seed

        # Create the environment
        self.env, self.env_cfg = task_registry.make_env(name=args.task, args=args)

        # Store attributes
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.device = self.env.device

    def reset(self, env_ids=None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment.

        Args:
            env_ids: IDs of environments to reset. If None, reset all environments.

        Returns:
            obs: Observation after reset
            info: Additional reset information
        """
        if env_ids is None:
            return self.env.reset()
        else:
            return self.env.reset(env_ids=env_ids)

    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment with the given actions.

        Args:
            actions: Actions to take in the environment

        Returns:
            obs: New observations
            rewards: Rewards for the action
            dones: Whether episodes are done
            infos: Additional information
        """
        result = self.env.step(actions)
        # The Isaac Gym environments return 5 values (obs, privileged_obs, rewards, dones, infos)
        # We only want to return 4 (obs, rewards, dones, infos)
        obs, _, rewards, dones, infos = result
        return obs, rewards, dones, infos

    def get_observations(self) -> torch.Tensor:
        """Get current observations from the environment.

        Returns:
            Current observations
        """
        return self.env.get_observations()

    def get_diffusion_observation(self) -> torch.Tensor:
        """Get observations formatted for diffusion policy.

        Returns:
            Observations in format suitable for diffusion policies
        """
        # Default implementation - simply return regular observations
        # Environments that support diffusion observations will override this method
        return self.get_observations()

    def render(self, mode="human"):
        """Render the environment.

        Args:
            mode: Rendering mode
        """
        # Isaac Gym environments typically don't implement a separate render method
        # as rendering is controlled via headless flag
        print("Rendering is controlled via the headless flag in Isaac Gym")
        return None

    def close(self):
        """Close the environment and clean up resources."""
        # Most Isaac Gym environments don't require explicit closing
        pass


# Example usage
if __name__ == "__main__":
    # Create a legged gym environment
    env = LeggedGymEnv(
        task_name="elspider_air_flat",  # Change to any registered task
        num_envs=4,
        headless=False,  # Set to False to visualize
        seed=42
    )

    print(f"Created environment with {env.num_envs} parallel environments")
    print(f"Observation space: {env.num_obs} dimensions")
    print(f"Action space: {env.num_actions} dimensions")

    # Reset the environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")

    # Run for a few steps with random actions
    for i in range(1000):
        # Generate random actions
        actions = torch.rand((env.num_envs, env.num_actions), device=env.device) * 2 - 1  # [-1, 1]

        # Step the environment
        obs, rewards, dones, infos = env.step(actions)

        # Print some information
        if i % 10 == 0:
            print(f"Step {i}, Reward: {rewards.mean().item()}")
        time.sleep(0.01)

    # Close the environment
    env.close()
    print("Environment closed successfully.")
