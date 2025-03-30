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
@click.option('-c', '--checkpoints', required=True, default=["legged_gym_cmp/legged_gym/logs/flat_elspider_air/exported/policies/policy_1.pt"], multiple=True, help="Paths to model checkpoints")
@click.option('-t', '--task_name', default="elspider_air_flat", help="Legged gym task name")
@click.option('-n', '--n_episodes', default=100, help="Number of episodes to collect per checkpoint")
@click.option('-e', '--episode_steps', default=400, help="Maximum steps per episode")
@click.option('-v', '--visualize', is_flag=True, help="Enable visualization")
@click.option('--headless', is_flag=True, help="Run in headless mode (no visualization)")
@click.option('--num_envs', default=40, help="Number of parallel environments to run")
@click.option('--seed', default=42, help="Random seed")
@click.option('--chunk_length', default=-1, help="Chunk length for zarr file, -1 for auto")
def main(output, checkpoints, task_name, n_episodes, episode_steps,
         visualize, headless, num_envs, seed, chunk_length):
    """Generate a dataset from legged gym environments using loaded checkpoints."""

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
        try:
            policy = load_policy_from_checkpoint(checkpoint_path, task_name)
            print(f"Loaded policy with shape - Obs: {policy.obs_dim}, Action: {policy.action_dim}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue

        # Track dimensions for this checkpoint
        checkpoint_meta = {
            'checkpoint_path': checkpoint_path,
            'obs_dim': env.num_obs,
            'action_dim': env.num_actions
        }

        # Initialize progress bar for this checkpoint
        checkpoint_progress = tqdm(total=n_episodes, desc=f"Checkpoint {os.path.basename(checkpoint_path)}")

        # Reset environment to start collecting episodes
        obs, _ = env.reset()

        # Track active episodes
        active_episodes = [{'obs': [], 'action': [], 'reward': []} for _ in range(env.num_envs)]
        steps_in_episode = np.zeros(env.num_envs, dtype=np.int32)

        # Continue until we've collected enough episodes for this checkpoint
        collected_episodes = 0

        while collected_episodes < n_episodes:
            # Get actions from policy
            with torch.no_grad():
                try:
                    # Make sure observations are on the right device
                    policy_device = next(policy.policy.parameters()).device if hasattr(policy, 'policy') else env.device

                    # Check if the observation needs to be moved to the policy's device
                    if obs.device != policy_device:
                        obs_on_device = obs.to(policy_device)
                    else:
                        obs_on_device = obs

                    # Get actions
                    actions = policy.predict_action(obs_on_device)

                    # Move actions back to the environment's device if needed
                    if actions.device != env.device:
                        actions = actions.to(env.device)

                except Exception as e:
                    print(f"Error during policy inference: {e}")
                    print(f"Observation shape: {obs.shape}, device: {obs.device}")
                    import traceback
                    traceback.print_exc()

                    # Fall back to random actions to continue
                    actions = torch.rand((env.num_envs, env.num_actions), device=env.device) * 2 - 1

            # Step the environment
            next_obs, rewards, dones, infos = env.step(actions)

            # Store data for each environment
            for i in range(env.num_envs):
                active_episodes[i]['obs'].append(obs[i].cpu().numpy())
                active_episodes[i]['action'].append(actions[i].cpu().numpy())
                active_episodes[i]['reward'].append([rewards[i].cpu().numpy()])  # Make reward a list to avoid zarr issue
                steps_in_episode[i] += 1

                # Check if episode is done (either by environment signal or max steps)
                if dones[i] or steps_in_episode[i] >= episode_steps:
                    # Only save episodes that have accumulated enough steps
                    if steps_in_episode[i] > 10:  # Minimum episode length threshold
                        steps = len(active_episodes[i]['obs'])
                        # Instead of storing the checkpoint path as a string, encode it as bytes
                        # This avoids the object_codec issue with zarr
                        episode_data = {
                            'obs': np.array(active_episodes[i]['obs']),
                            'action': np.array(active_episodes[i]['action']),
                            'reward': np.array(active_episodes[i]['reward']),
                            # Store numeric metadata directly
                            'obs_dim': np.full(steps, env.num_obs, dtype=np.int32),
                            'action_dim': np.full(steps, env.num_actions, dtype=np.int32),
                            # For the checkpoint path, just store the filename to avoid codec issues
                            'checkpoint_name': np.array([os.path.basename(checkpoint_path)] * steps, dtype=np.string_)
                        }
                        buffer.add_episode(episode_data)

                        collected_episodes += 1
                        checkpoint_progress.update(1)
                        global_progress.update(1)

                    # Reset this environment's episode
                    active_episodes[i] = {'obs': [], 'action': [], 'reward': []}
                    steps_in_episode[i] = 0

                    # If we've collected enough episodes, break the loop
                    if collected_episodes >= n_episodes:
                        break

            # Update obs for next step
            obs = next_obs

            # If we've collected enough episodes, break the loop
            if collected_episodes >= n_episodes:
                break

        checkpoint_progress.close()

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


def load_policy_from_checkpoint(checkpoint_path, task_name):
    """
    Load a policy from a checkpoint file using the legged gym task registry system.
    This mimics the policy loading approach used in play.py.
    """
    try:
        # Get the directory containing the checkpoint
        checkpoint_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
        experiment_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))

        # Create dummy args for loading
        from legged_gym.utils import get_known_args
        args = get_known_args()
        args.task = task_name  # This will be overridden by the checkpoint
        args.headless = True
        args.num_envs = 1  # Just need one env for loading the policy

        # We need a dummy environment to load the policy
        # We'll create a minimal env just for policy loading
        from legged_gym.utils import task_registry
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

        # Override training config to load from the specified checkpoint
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = checkpoint_dir

        # Extract checkpoint number from filename if it follows the pattern model_X.pt
        checkpoint_name = os.path.basename(checkpoint_path)
        if checkpoint_name.startswith("model_") and checkpoint_name.endswith(".pt"):
            checkpoint_num = int(checkpoint_name[6:-3])
            train_cfg.runner.checkpoint = checkpoint_num
        elif checkpoint_name == "policy_1.pt":
            # This is an exported policy, which requires a different loading approach
            return load_exported_policy(checkpoint_path)

        # Create a temporary env just for loading the policy
        temp_env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

        # Create the algorithm runner
        ppo_runner, _ = task_registry.make_alg_runner(
            env=temp_env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
            log_root=None  # Avoid creating logs during dataset generation
        )

        # Get the inference policy
        policy = ppo_runner.get_inference_policy(device=temp_env.device)

        # Set dimensions for reference
        policy.obs_dim = temp_env.num_obs
        policy.action_dim = temp_env.num_actions

        print(f"Successfully loaded policy from checkpoint {checkpoint_path}")
        return policy

    except Exception as e:
        print(f"Error loading policy from checkpoint: {str(e)}")
        raise e


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
            def __init__(self, jit_policy, action_dim=18, obs_dim=64):
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
