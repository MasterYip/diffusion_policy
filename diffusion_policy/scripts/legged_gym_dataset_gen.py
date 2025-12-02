if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
import numpy as np
import isaacgym
import torch
import json
from tqdm import tqdm
import time
from diffusion_policy.common.replay_buffer import ReplayBuffer

# Import the legged gym environment wrapper
from diffusion_policy.env.legged_gym.legged_gym_env import LeggedGymEnv


@click.command()
@click.option('-o', '--output', required=True, default="diffusion_policy/data/legged_gym/elspider_dataset.zarr", help="Path to save dataset (e.g., data/legged_gym/anymal_dataset.zarr)")
@click.option('-c', '--checkpoints', required=True, default=["source_ckpts/elspider_air_flat.pt"], multiple=True, help="Paths to expert policy checkpoints")
@click.option('-t', '--task_name', default="elspider_air_flat", help="Legged gym task name")
@click.option('-n', '--n_episodes', default=1000, help="[unused]Number of episodes to collect per checkpoint")
@click.option('-e', '--episode_steps', default=500, help="[unused]Maximum steps per episode")
@click.option('-v', '--visualize', is_flag=True, help="Enable visualization")
@click.option('--headless', is_flag=True, help="Run in headless mode (no visualization)")
@click.option('--num_envs', default=100, help="Number of parallel environments to run")
@click.option('--seed', default=42, help="Random seed")
@click.option('--chunk_length', default=-1, help="Chunk length for zarr file, -1 for auto")
@click.option('--n_obs_steps', default=8, help="Number of observation steps for history")
@click.option('--len_to_save', default=500000, help="Total length of data to save")
def main(output, checkpoints, task_name, n_episodes, episode_steps,
         visualize, headless, num_envs, seed, chunk_length, n_obs_steps, len_to_save):
    """Generate a dataset from legged gym environments using expert policies."""

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize replay buffer
    buffer = ReplayBuffer.create_empty_numpy()

    # Store metadata about the dataset
    metadata = {
        'task_name': task_name,
        'n_episodes_per_checkpoint': n_episodes,
        'episode_steps': episode_steps,
        'checkpoints': checkpoints,
        'seed': seed,
        'creation_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Calculate total episodes across all checkpoints
    total_episodes = len(checkpoints) * n_episodes

    # Progress tracking
    episode_count = 0
    global_progress = tqdm(total=total_episodes, desc="Total progress")

    # Create environment
    env = LeggedGymEnv(
        task_name=task_name,
        num_envs=num_envs,
        headless=not visualize and headless,
        seed=seed
    )

    # Process each checkpoint
    for checkpoint_path in checkpoints:
        print(f"Processing checkpoint: {checkpoint_path}")

        # Load policy from checkpoint
        policy = load_policy_from_checkpoint(checkpoint_path, task_name, env=env.env)
        print(f"Loaded policy with shape - Obs: {policy.obs_dim}, Action: {policy.action_dim}")

        device = env.device
        dtype = torch.float32
        
        # Initialize state and action history (matching cyber_runner exactly)
        history = n_obs_steps
        state_history = torch.zeros((env.num_envs, history+1, env.num_obs), dtype=dtype, device=device)
        action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=dtype, device=device)

        # Reset environment and initialize state history
        obs, _ = env.reset()
        state_history[:, :, :] = env.get_diffusion_observation().to(device)[:, None, :]

        # Initialize episode recording (matching cyber_runner structure)
        recorded_obs_episode = np.zeros((env.num_envs, env.max_episode_length+2, env.num_obs))
        recorded_acs_episode = np.zeros((env.num_envs, env.max_episode_length+3, env.num_actions))

        # Data collection variables
        recorded_obs = []
        recorded_acs = []
        episode_ends = []
        saved_idx = 0
        step_count = 0

        # Initialize progress bar
        pbar = tqdm(total=len_to_save, desc=f"Collecting data from {os.path.basename(checkpoint_path)}")

        while saved_idx < len_to_save:
            # Get current observation for recording
            single_obs_dict = {"obs": state_history[:, -1, :].to(device)}

            # Get expert actions (matching cyber_runner)
            with torch.no_grad():
                expert_action = policy.predict_action(obs.detach())
                action = expert_action[:, None, :]

            # Record obs and actions BEFORE stepping (like cyber_runner)
            curr_idx = np.all(recorded_obs_episode == 0, axis=-1).argmax(axis=-1)
            recorded_obs_episode[np.arange(env.num_envs), curr_idx, :] = single_obs_dict["obs"].cpu().detach().numpy()
            recorded_acs_episode[np.arange(env.num_envs), curr_idx, :] = expert_action.cpu().detach().numpy()

            # Step environment (matching cyber_runner's multi-step approach)
            n_action_steps = action.shape[1]
            for i in range(n_action_steps):
                action_step = action[:, i, :]
                obs, rews, done, infos = env.step(action_step)

                # Update state and action history (matching cyber_runner)
                state_history = torch.roll(state_history, shifts=-1, dims=1)
                action_history = torch.roll(action_history, shifts=-1, dims=1)

                state_history[:, -1, :] = env.get_diffusion_observation().to(device)
                single_obs_dict = {"obs": state_history[:, -1, :].to(device)}

                step_count += 1

            # Handle episode terminations (matching cyber_runner exactly)
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(1).int()
            if len(env_ids) > 0:
                # Reset state and action history
                state_history[env_ids, :, :] = single_obs_dict["obs"][env_ids].to(state_history.device)[:, None, :]
                action_history[env_ids, :, :] = 0.0

                # Process completed episodes (matching cyber_runner's saving logic)
                for i in range(len(env_ids)):
                    env_idx = env_ids[i]
                    epi_len = np.all(recorded_obs_episode[env_idx] == 0, axis=-1).argmax(axis=-1)
                    if epi_len == 0:
                        epi_len = recorded_acs_episode.shape[1]
                    
                    # Only save episodes longer than 400 steps (like cyber_runner)
                    if epi_len > 400:
                        recorded_obs.append(np.copy(recorded_obs_episode[env_idx, :epi_len]))
                        recorded_acs.append(np.copy(recorded_acs_episode[env_idx, :epi_len]))
                        saved_idx += epi_len
                        episode_ends.append(saved_idx)
                        
                        print(f"Saved episode with length {epi_len}, total saved_idx: {saved_idx}")
                        pbar.update(epi_len)

                    # Reset episode recording
                    recorded_obs_episode[env_idx] = 0
                    recorded_acs_episode[env_idx] = 0

            # Break if we've collected enough data
            if saved_idx >= len_to_save:
                break

        pbar.close()

        # Process collected data into zarr format
        if recorded_obs and recorded_acs:
            print("Converting collected data to dataset format...")
            
            # Concatenate all episodes
            all_obs = np.concatenate(recorded_obs, axis=0)
            all_actions = np.concatenate(recorded_acs, axis=0)
            episode_ends = np.array(episode_ends)

            print(f"Total observations: {all_obs.shape}")
            print(f"Total actions: {all_actions.shape}")
            print(f"Total episodes: {len(episode_ends)}")

            # Convert to episode format for ReplayBuffer
            episode_start = 0
            for episode_end in episode_ends:
                episode_length = episode_end - episode_start
                
                episode_obs = all_obs[episode_start:episode_end]
                episode_actions = all_actions[episode_start:episode_end]
                episode_rewards = np.ones((episode_length, 1))  # Placeholder rewards
                
                episode_data = {
                    'obs': episode_obs,
                    'action': episode_actions,
                    'reward': episode_rewards,
                    'obs_dim': np.full(episode_length, env.num_obs, dtype=np.int32),
                    'action_dim': np.full(episode_length, env.num_actions, dtype=np.int32),
                    'checkpoint_name': np.array([os.path.basename(checkpoint_path)] * episode_length, dtype=np.string_)
                }
                
                buffer.add_episode(episode_data)
                episode_start = episode_end

    global_progress.close()

    # Save metadata separately as JSON
    with open(os.path.join(output_dir, os.path.basename(output).replace('.zarr', '_metadata.json')), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save the dataset with proper codecs
    print(f"Saving dataset to {output}")
    import numcodecs
    buffer.save_to_path(
        output,
        chunk_length=chunk_length,
        # Add specific options for zarr arrays with object dtypes
        zarr_arr_opts={
            'checkpoint_name': {
                'object_codec': numcodecs.VLenUTF8()
            }
        }
    )

    print(f"Dataset generation complete. Total episodes: {buffer.n_episodes}")


def load_policy_from_checkpoint(checkpoint_path, task_name, env=None):
    """
    Load a policy from a checkpoint file using a direct approach without full runner instantiation.
    """
    try:
        print("Try Loading JIT checkpoint...")
        # This is an exported policy, which requires a different loading approach
        return load_exported_policy(checkpoint_path)
    except Exception as e:
        print(f"Not a JIT checkpoint, loading normally...")

    # Try loading as a simple policy first
    try:
        print("Try loading as simple policy...")
        policy = torch.load(checkpoint_path, map_location="cuda:0")

        # Create wrapper for consistency
        class RegularPolicyWrapper:
            def __init__(self, policy, action_dim=None, obs_dim=None):
                self.policy = policy
                # Try to infer dimensions from the policy if possible
                if hasattr(policy, 'num_actions'):
                    self.action_dim = policy.num_actions
                elif action_dim is not None:
                    self.action_dim = action_dim
                else:
                    self.action_dim = env.num_actions if env else 18  # Default fallback
                
                if hasattr(policy, 'num_obs'):
                    self.obs_dim = policy.num_obs
                elif obs_dim is not None:
                    self.obs_dim = obs_dim
                else:
                    self.obs_dim = env.num_obs if env else 66  # Default fallback
                
                # Determine device
                if hasattr(policy, 'parameters'):
                    try:
                        self.device = next(policy.parameters()).device
                    except StopIteration:
                        self.device = torch.device("cuda:0")
                else:
                    self.device = torch.device("cuda:0")
                
                print(f"Policy loaded and running on device: {self.device}")
                
                # Set to eval mode if possible
                if hasattr(policy, 'eval'):
                    policy.eval()

            def predict_action(self, obs):
                # Ensure observations are on the same device as the policy
                if isinstance(obs, torch.Tensor):
                    if obs.device != self.device:
                        obs = obs.to(self.device)
                    
                    # Try different policy call methods
                    if hasattr(self.policy, 'act_inference'):
                        return self.policy.act_inference(obs)
                    elif hasattr(self.policy, 'act'):
                        return self.policy.act(obs)
                    elif callable(self.policy):
                        return self.policy(obs)
                    else:
                        raise AttributeError("Policy has no callable method (act_inference, act, or __call__)")
                else:
                    # Convert to tensor if not already
                    obs_tensor = torch.as_tensor(obs, device=self.device)
                    return self.predict_action(obs_tensor)

            def __call__(self, obs):
                return self.predict_action(obs)

            def reset(self):
                # Reset the policy state if it's stateful
                if hasattr(self.policy, 'reset'):
                    self.policy.reset()

        # Create the wrapped policy
        wrapped_policy = RegularPolicyWrapper(policy, env.num_actions if env else None, env.num_obs if env else None)
        print(f"Successfully loaded simple policy from {checkpoint_path}")
        return wrapped_policy
        
    except Exception as simple_load_error:
        print(f"Not a simple policy, trying checkpoint loading: {simple_load_error}")



def load_exported_policy(checkpoint_path):
    """
    Load an exported policy (JIT script) directly.
    """
    try:
        # Load the JIT policy and move it to the GPU
        policy = torch.jit.load(checkpoint_path, map_location='cuda:0')

        # Create a wrapper to make the interface consistent
        class JitPolicyWrapper:
            # FIXME: action_dim and obs_dim should be inferred from the policy
            def __init__(self, jit_policy, action_dim=18, obs_dim=66):
                self.policy = jit_policy
                self.action_dim = action_dim
                self.obs_dim = obs_dim
                self.device = next(self.policy.parameters()).device
                print(f"Policy loaded and running on device: {self.device}")

            def predict_action(self, obs):
                # Ensure observations are on the same device as the policy
                if isinstance(obs, torch.Tensor):
                    # Make sure the input is the right shape and on the right device
                    if obs.device != self.device:
                        obs = obs.to(self.device)

                    # For JIT policies, they sometimes expect specific input formats
                    # If it's a 2D tensor (batch_size, obs_dim), we need to handle it
                    if len(obs.shape) == 2:
                        # The policy might need a sequence dimension
                        return self.policy(obs)
                    else:
                        # Try different formats if needed
                        return self.policy(obs)
                else:
                    # Convert to tensor if not already
                    obs_tensor = torch.as_tensor(obs, device=self.device)
                    return self.policy(obs_tensor)

            def __call__(self, obs):
                return self.predict_action(obs)

            def reset(self):
                # Reset the policy state if it's stateful (e.g., has memory)
                if hasattr(self.policy, 'reset_memory'):
                    self.policy.reset_memory()

        # Create the wrapped policy
        wrapped_policy = JitPolicyWrapper(policy)

        print(f"Successfully loaded exported JIT policy from {checkpoint_path}")
        return wrapped_policy

    except Exception as e:
        print(f"Error loading exported policy: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    main()
