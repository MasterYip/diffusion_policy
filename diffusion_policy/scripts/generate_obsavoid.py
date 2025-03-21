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
import random
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.obsavoid.obsavoid_env import randpath_bound_env, sine_bound_env

import time


@click.command()
@click.option('-o', '--output', required=True, default="data/obsavoid/obsavoid_replay.zarr")
@click.option('-n', '--n_episodes', default=250)
@click.option('-e', '--episode_steps', default=200)
@click.option('-c', '--chunk_length', default=-1)
@click.option('--ctrl_mode', default='acc', type=click.Choice(['dy', 'y', 'acc']))
@click.option('-v', '--visualize', default=True)
def main(output, n_episodes, episode_steps, chunk_length, ctrl_mode, visualize, vis_interval=100):

    buffer = ReplayBuffer.create_empty_numpy()

    for i in tqdm(range(n_episodes)):
        env = randpath_bound_env(visualize,
                                 y=None,
                                 v=None,
                                 env_step=0.01)
        obs_history = list()
        action_history = list()
        for i in range(episode_steps):
            observation = env.get_observation()
            obs_history.append(observation)
            # reward = env.get_reward()
            # rewards.append(reward)

            if ctrl_mode == 'dy':
                action = env.get_action_dy()
                action_history.append(action)
            elif ctrl_mode == 'y':
                action = env.get_action_y()
                action_history.append(action)
            elif ctrl_mode == 'acc':
                action = env.get_action()
                action_history.append(action)
            else:
                raise ValueError("Invalid ctrl_mode")
            print("action:", action[0])
            env.step_env(acc=env.get_action()[0], vis=False)
            # Visualize (per 100 steps)
            if (visualize and i % vis_interval == 0):
                env.vis_step()
        env.end()

        obs_history = np.array(obs_history)
        action_history = np.array(action_history)
        episode = {
            'obs': obs_history,
            'action': action_history
        }
        buffer.add_episode(episode)

    buffer.save_to_path(output, chunk_length=chunk_length)


if __name__ == '__main__':
    main()
