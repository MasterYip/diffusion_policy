from math import exp
from typing import Optional, Callable, Dict, Tuple
from collections import namedtuple
from omegaconf import DictConfig
import torch
from einops import rearrange, reduce
from torch import nn
from torch.nn import functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer

from diffusion_policy.model.diffusion.rolling_diffusion import RollingDiffusion
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
import numpy as np

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])


def exp_noise_mask(horizon, max_noise_level, sigma=2.0, pad_zero=1,
                   dtype=torch.int64):
    zeros = torch.zeros(pad_zero)
    len_exp = horizon - pad_zero
    exps = torch.tensor([exp((k-len_exp)*sigma / len_exp) for k in range(len_exp)])
    exps = ((exps - exp(-sigma)) / (1.0 - exp(-sigma)) * max_noise_level)
    return torch.cat([zeros, exps]).to(dtype)


class RollDiffTransformerLowdimPolicy(BaseLowdimPolicy):

    def __init__(self,
                 model: RollingDiffusion,
                 horizon,
                 obs_dim,
                 action_dim,
                 n_action_steps,
                 n_obs_steps,
                 max_noise_level=None,
                 obs_as_cond=True,
                 pred_action_steps_only=True,
                 # parameters passed to step
                 **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        # FIXME: temp 1
        self.frame_stack = 1

        if max_noise_level is None:
            raise ValueError("max_noise_level must be provided")
        self.max_noise_level = max_noise_level

        # Cached Trajectory for rolling diffusion
        self.trajectory = None

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        # Get transformer optimizer in rolling diffusion
        return self.model.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas))

    # Function for rolling diff
    def _generate_noise_levels(self, batch_size: int, horizon: int) -> torch.Tensor:
        """ Generate rand noise levels for traning """
        noise_levels = torch.randint(0, self.max_noise_level, (batch_size, horizon), device=self.device)
        return noise_levels

    def get_noise_mask(self, window=20, zero_noise_pad=20, uncertainty_scale=1):
        """ Linearly increase noise level """
        zeros = torch.zeros(zero_noise_pad, dtype=torch.int32)
        increase = torch.tensor([uncertainty_scale*(k+1) for k in range(window-zero_noise_pad)], dtype=torch.int32)
        return torch.cat([zeros, increase]).to(self.device)

    # DEPRECATED
    def get_last_noise_mask(self, window=20, zero_noise_pad=20, uncertainty_scale=1):
        """ Shift 1 step back """
        mask = self.get_noise_mask(window, zero_noise_pad, uncertainty_scale)
        return torch.cat([mask[1:], mask[-1].unsqueeze(0)]).to(self.device)

    def shifted_noise_mask(self, noise_mask):
        return torch.cat([noise_mask[:, 1:], noise_mask[:, -1].unsqueeze(1)], dim=1).to(self.device)

    def get_const_noise_mask(self, window=20, noise_level=1):
        return torch.tensor([noise_level for _ in range(window)]).unsqueeze(0).long().to(self.device)

    def init_trajectory(self, batch_size,  horizon: int, action_dim: int):
        # start = self.make_bundle()
        plan_horizon = np.ceil(horizon / self.frame_stack).astype(int)

        chunk = torch.randn((batch_size, plan_horizon, action_dim), device=self.device)
        # chunk = torch.clamp(chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)

        # pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        # init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        # plan_traj = torch.cat([init_token, chunk, pad], 0)
        plan_traj = chunk
        # (B,Ta,Da)

        # Initialize noise levels
        # self.from_noise_levels = self.get_last_noise_mask(
        #     plan_horizon, 1, self.max_noise_level/plan_horizon).repeat(batch_size, 1).long().to(self.device)
        # self.to_noise_levels = self.get_noise_mask(plan_horizon, 1, self.max_noise_level /
        #                                            plan_horizon).repeat(batch_size, 1).long().to(self.device)
        self.to_noise_levels = exp_noise_mask(plan_horizon, self.max_noise_level,
                                              sigma=2.0, pad_zero=1).repeat(batch_size, 1).to(self.device)
        self.from_noise_levels = self.shifted_noise_mask(self.to_noise_levels)
        return plan_traj

    def ddim_step(self, plan_traj, from_noise_levels=None, to_noise_levels=None, condition=None):
        """ One step denoising for feedback control """
        if from_noise_levels is None:
            from_noise_levels = self.from_noise_levels
        if to_noise_levels is None:
            to_noise_levels = self.to_noise_levels
        # Fix the first token
        # plan_traj[:, 1:] = self.model.sample_step(
        #     plan_traj, condition, from_noise_levels, to_noise_levels, guidance_fn=None
        # )[:, 1:]
        plan_traj[:, 1:] = self.model.ddim_sample_step(
            x=plan_traj,
            external_cond=condition,
            curr_noise_level=from_noise_levels,
            next_noise_level=to_noise_levels,
            guidance_fn=None,
        )[:, 1:]

    # interface (copied from diffusion policy)

    def shift_trajectory(self, plan_traj, append_noise=False):
        """ Shift 1 step back """
        batch_size = plan_traj.shape[0]
        if append_noise:
            chunk = torch.randn((batch_size, 1, self.action_dim), device=self.device)
        else:
            chunk = plan_traj[:, -1:, :]
        return torch.cat([plan_traj[:, 1:, :], chunk], dim=1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: (B, To, Do)  # To=observation horizon
        return:
            action: (B, Ta, Da)  # Ta=action horizon
        """
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict  # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # # build input
        # device = self.device
        # dtype = self.dtype

        # handle different ways of passing observation
        cond = None  # Conditions
        assert self.obs_as_cond  # only support obs_as_cond
        if self.obs_as_cond:
            cond = nobs[:, :To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            # initialize trajectory if not exist
            if self.trajectory is None or self.trajectory.shape != shape:
                self.trajectory = self.init_trajectory(*shape)
                for i in range(self.horizon):
                    from_noise_levels = self.get_const_noise_mask(
                        shape[1], self.max_noise_level*(1 - i/self.horizon)-1).repeat(B, 1)
                    to_noise_levels = self.get_const_noise_mask(
                        shape[1], self.max_noise_level*(1 - (i+1)/self.horizon)-1).repeat(B, 1)
                    self.ddim_step(self.trajectory,
                                   from_noise_levels=from_noise_levels,
                                   to_noise_levels=to_noise_levels, condition=cond)

        self.ddim_step(self.trajectory,
                       from_noise_levels=self.from_noise_levels,
                       to_noise_levels=self.to_noise_levels,
                       condition=cond)

        self.trajectory = self.shift_trajectory(self.trajectory)

        # Unnormalize action
        nsample = self.trajectory
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict:
            obs: (B, To, Do)  # To=observation horizon
        return:
            action: (B, Ta, Da)  # Ta=action horizon
        """
        # 数据标准化
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:, :self.n_obs_steps, :]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            raise NotImplementedError()

        # generate impainting mask
        # if self.pred_action_steps_only:
        #     condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        # else:
        #     condition_mask = self.mask_generator(trajectory.shape)

        rand_noise_levels = self._generate_noise_levels(trajectory.shape[0], trajectory.shape[1])

        x_pred, loss = self.model(
            x=trajectory,
            external_cond=cond,
            noise_levels=rand_noise_levels,
        )

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss
