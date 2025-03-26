from typing import Optional, Callable, Dict
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from diffusion_policy.model.diffusion.transformer_for_rolling_diff import TransformerForRollingDiffusion
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
import numpy as np

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])


class RollDiffTransformerLowdimPolicy(BaseLowdimPolicy):
    # Special thanks to lucidrains for the implementation of the base RollDiffTransformerLowdimPolicy model
    # https://github.com/lucidrains/denoising-diffusion-pytorch

    def __init__(
        self,
        x_shape: torch.Size,
        cfg: DictConfig,
        #=== model parameters ===
        model: TransformerForRollingDiffusion,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_cond=True,
        pred_action_steps_only=True,
        # parameters passed to step
        **kwargs
    ):
        super().__init__()

        self.x_shape = x_shape
        self.timesteps = cfg.timesteps  # Total timesteps
        self.sampling_timesteps = cfg.sampling_timesteps  # Sampling timesteps for DDIM
        self.beta_schedule = cfg.beta_schedule
        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        self.objective = cfg.objective
        self.use_fused_snr = cfg.use_fused_snr
        self.snr_clip = cfg.snr_clip
        self.cum_snr_decay = cfg.cum_snr_decay
        self.ddim_sampling_eta = cfg.ddim_sampling_eta  # DDIM sampling eta
        self.clip_noise = cfg.clip_noise
        self.arch = cfg.architecture
        self.stabilization_level = cfg.stabilization_level

        self.model = model
        self._build_buffer()

    # Function for rolling diff

    def init_trajectory(self, horizon: int):
        batch_size = 1
        # start = self.make_bundle()
        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)

        print("self.x_stacked_shape:", self.x_stacked_shape)
        chunk = torch.randn((plan_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        chunk = torch.clamp(chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
        # pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        # init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        # plan_traj = torch.cat([init_token, chunk, pad], 0)
        plan_traj = chunk
        print("plan_traj:", plan_traj.shape)

        self.from_noise_levels = self.get_last_noise_mask(plan_tokens)
        self.to_noise_levels = self.get_noise_mask(plan_tokens)
        return plan_traj

    def ddim_step(self, plan_traj, from_noise_levels=None, to_noise_levels=None, condition=None):
        """ One step denoising for feedback control """
        if from_noise_levels is None:
            from_noise_levels = self.from_noise_levels
        if to_noise_levels is None:
            to_noise_levels = self.to_noise_levels
        # Fix the first token
        plan_traj[1:] = self.sample_step(
            plan_traj, condition, from_noise_levels, to_noise_levels, guidance_fn=None
        )[1:]

    # interface (copied from diffusion policy)
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: (B, To, Do)  # To=observation horizon
        return: 
            action: (B, Ta, Da)  # Ta=action horizon
        """
        # 数据标准化
        nobs = self.normalizer.normalize(obs_dict['obs'], 'observations')
        cond = nobs[:, :self.n_obs_steps, :]  # (B, n_obs_steps, Do)
        
        # 初始化噪声轨迹
        batch_size = cond.shape[0]
        trajectory = torch.randn(
            (batch_size, self.n_action_steps, self.action_dim),
            device=cond.device
        )
        
        # DDIM采样过程
        timesteps = torch.linspace(0, self.timesteps-1, self.sampling_timesteps, device=cond.device).long()
        
        for t in reversed(range(0, self.sampling_timesteps)):
            ts = torch.full((batch_size,), t, device=cond.device, dtype=torch.long)
            
            # 生成带噪声的轨迹
            noisy_trajectory = self.q_sample(
                x_start=trajectory,
                t=ts,
                noise=torch.randn_like(trajectory)
            )
            
            # 模型预测
            model_pred = self.model_predictions(
                x=noisy_trajectory,
                t=ts,
                external_cond=cond
            )
            
            # 更新轨迹
            trajectory = model_pred.pred_x_start.detach()
        
        # 反标准化输出
        unnorm_action = self.normalizer.unnormalize(trajectory, 'actions')
        return {'action': unnorm_action}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 数据标准化
        nobs = self.normalizer.normalize(batch['obs'], 'observations')
        naction = self.normalizer.normalize(batch['action'], 'actions')
        
        # 构建条件
        cond = nobs[:, :self.n_obs_steps, :]  # (B, n_obs_steps, Do)
        
        # 生成噪声时间步
        batch_size = cond.shape[0]
        timesteps = torch.randint(
            0, self.timesteps,
            (batch_size,), device=cond.device
        ).long()
        
        # 生成噪声
        noise = torch.randn_like(naction)
        noisy_actions = self.q_sample(
            x_start=naction,
            t=timesteps,
            noise=noise
        )
        
        # 模型预测
        model_pred = self.model_predictions(
            x=noisy_actions,
            t=timesteps,
            external_cond=cond
        )
        
        # 计算损失（支持多种预测目标）
        target = {
            'pred_noise': noise,
            'pred_x0': naction,
            'pred_v': self.predict_v(naction, timesteps, noise)
        }[self.objective]
        
        loss = F.mse_loss(model_pred.model_out, target)
        return loss
    
    # def set_normalizer(self, normalizer: LinearNormalizer):
    #     self.normalizer.load_state_dict(normalizer.state_dict())

    # def get_optimizer(
    #     self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    # ) -> torch.optim.Optimizer:
    #     return self.model.configure_optimizers(
    #         weight_decay=weight_decay,
    #         learning_rate=learning_rate,
    #         betas=tuple(betas))