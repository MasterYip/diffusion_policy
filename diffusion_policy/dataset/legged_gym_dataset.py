from typing import Dict, Optional, List, Tuple, Any
import torch
import numpy as np
import copy
import json
from pathlib import Path

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class LeggedGymDataset(BaseLowdimDataset):
    """
    Dataset for legged gym environments that supports variable input/output dimensions
    from different checkpoints.
    """

    def __init__(self,
                 zarr_path: str,
                 horizon: int = 1,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 obs_key: str = 'obs',
                 action_key: str = 'action',
                 reward_key: str = 'reward',
                 checkpoint_filter: Optional[List[str]] = None,
                 seed: int = 42,
                 val_ratio: float = 0.0,
                 max_train_episodes: Optional[int] = None):
        """
        Initialize the dataset.

        Args:
            zarr_path: Path to the zarr dataset
            horizon: Number of timesteps to include in each sample
            pad_before: Number of timesteps to pad before the sequence
            pad_after: Number of timesteps to pad after the sequence
            obs_key: Key for observations in the dataset
            action_key: Key for actions in the dataset
            reward_key: Key for rewards in the dataset
            checkpoint_filter: List of checkpoint paths to include, or None to include all
            seed: Random seed for validation split
            val_ratio: Ratio of data to use for validation
            max_train_episodes: Maximum number of episodes to use for training
        """
        super().__init__()

        # Load the data
        keys = [obs_key, action_key, reward_key, 'checkpoint_meta']
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        # Try to load metadata if available
        metadata_path = Path(zarr_path).parent / f"{Path(zarr_path).stem}_metadata.json"
        self.metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata from {metadata_path}: {e}")

        # Filter episodes by checkpoint if specified
        filtered_idxs = list(range(self.replay_buffer.n_episodes))
        if checkpoint_filter is not None:
            # Get list of checkpoints for each episode
            filtered_idxs = []
            for i in range(self.replay_buffer.n_episodes):
                episode = self.replay_buffer.get_episode(i)
                checkpoint = episode['checkpoint_meta']['checkpoint_path']
                if any(cf in checkpoint for cf in checkpoint_filter):
                    filtered_idxs.append(i)

            print(f"Filtered to {len(filtered_idxs)}/{self.replay_buffer.n_episodes} episodes matching checkpoint filter")

        # Create episode mask for training/validation split
        all_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
        all_mask[filtered_idxs] = True

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)

        # Only use episodes that pass both filters for validation
        val_mask = val_mask & all_mask

        # Training episodes are those that pass the checkpoint filter but aren't in validation
        train_mask = all_mask & ~val_mask

        # Downsample training episodes if max_train_episodes is specified
        if max_train_episodes is not None:
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed)

        # Create sampler for training data
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        # Store attributes
        self.obs_key = obs_key
        self.action_key = action_key
        self.reward_key = reward_key
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Cache dimensions from first episode for quick access
        # (individual episodes may have different dimensions)
        first_ep_idx = np.where(train_mask)[0][0] if np.any(train_mask) else 0
        first_ep = self.replay_buffer.get_episode(first_ep_idx)
        self.example_obs_shape = first_ep[obs_key].shape[1:]
        self.example_action_shape = first_ep[action_key].shape[1:]

    def get_validation_dataset(self) -> 'LeggedGymDataset':
        """Create a validation dataset with the same parameters."""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        val_set.train_mask = self.val_mask
        val_set.val_mask = self.val_mask  # Both masks are the same for validation set
        return val_set

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        """Get a normalizer fit to the training data."""
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Get all actions from the dataset."""
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.sampler)

    def _sample_to_data(self, sample) -> Dict[str, np.ndarray]:
        """Convert a sample to a data dictionary."""
        data = {
            'obs': sample[self.obs_key],       # T, D_o
            'action': sample[self.action_key],  # T, D_a
        }

        # Add reward if available
        if self.reward_key in sample:
            data['reward'] = sample[self.reward_key]  # T

        # Add checkpoint metadata if available
        if 'checkpoint_meta' in sample:
            data['checkpoint_meta'] = sample['checkpoint_meta']

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        # Convert numpy arrays to PyTorch tensors
        torch_data = dict_apply(data, torch.from_numpy)

        # Add metadata information for debugging and tracking
        if 'checkpoint_meta' in sample:
            # Store as string to avoid issues with tensor conversion
            torch_data['checkpoint_path'] = sample['checkpoint_meta']['checkpoint_path']
            torch_data['obs_dim'] = sample['checkpoint_meta']['obs_dim']
            torch_data['action_dim'] = sample['checkpoint_meta']['action_dim']

        return torch_data

    def get_episode_dimensions(self) -> List[Tuple[int, int]]:
        """
        Get the observation and action dimensions for each episode in the dataset.

        Returns:
            List of (obs_dim, action_dim) tuples for each episode
        """
        dimensions = []
        for i in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(i)
            if 'checkpoint_meta' in episode:
                obs_dim = episode['checkpoint_meta']['obs_dim']
                action_dim = episode['checkpoint_meta']['action_dim']
                dimensions.append((obs_dim, action_dim))
            else:
                # Fallback to using shape if metadata not available
                obs_dim = episode[self.obs_key].shape[1]
                action_dim = episode[self.action_key].shape[1]
                dimensions.append((obs_dim, action_dim))

        return dimensions

    def get_unique_dimensions(self) -> List[Tuple[int, int]]:
        """
        Get the unique observation and action dimension pairs in the dataset.

        Returns:
            List of unique (obs_dim, action_dim) tuples
        """
        all_dims = self.get_episode_dimensions()
        return list(set(all_dims))

    def filter_by_dimensions(self, obs_dim: int, action_dim: int) -> 'LeggedGymDataset':
        """
        Create a new dataset containing only episodes with matching dimensions.

        Args:
            obs_dim: Observation dimension to filter for
            action_dim: Action dimension to filter for

        Returns:
            New dataset with only matching episodes
        """
        filtered_dataset = copy.copy(self)

        # Create a mask for episodes with matching dimensions
        dimension_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)

        for i in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(i)
            episode_obs_dim = episode['checkpoint_meta']['obs_dim']
            episode_action_dim = episode['checkpoint_meta']['action_dim']

            if episode_obs_dim == obs_dim and episode_action_dim == action_dim:
                dimension_mask[i] = True

        # Update masks
        filtered_train_mask = self.train_mask & dimension_mask
        filtered_val_mask = self.val_mask & dimension_mask

        # Create new samplers
        filtered_dataset.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=filtered_train_mask
        )

        filtered_dataset.train_mask = filtered_train_mask
        filtered_dataset.val_mask = filtered_val_mask

        print(f"Filtered to {np.sum(filtered_train_mask)}/{np.sum(self.train_mask)} training episodes and "
              f"{np.sum(filtered_val_mask)}/{np.sum(self.val_mask)} validation episodes "
              f"with obs_dim={obs_dim}, action_dim={action_dim}")

        return filtered_dataset
