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
from diffusion_policy.env.obsavoid.obsavoid_env import randpath_bound_env


@click.command()
@click.option('-o', '--output', required=True, default="data/obsavoid/obsavoid_replay.zarr")
@click.option('-n', '--n_episodes', default=1000)
@click.option('-e', '--episode_steps', default=1000)
@click.option('-c', '--chunk_length', default=-1)
def main(output, n_episodes, episode_steps, chunk_length):

    buffer = ReplayBuffer.create_empty_numpy()
    
    for i in tqdm(range(n_episodes)):
        env = randpath_bound_env(False, 
                             y=random.uniform(-1, 1), 
                             v=random.uniform(-1, 1), 
                             env_step=0.01)
        obs_history = list()
        action_history = list()
        for _ in range(episode_steps):
            observation = env.get_observation()
            action = env.get_action()
            # reward = env.get_reward()
            obs_history.append(observation)
            action_history.append(action)
            # rewards.append(reward)
            env.step_env(acc=action[0])
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
