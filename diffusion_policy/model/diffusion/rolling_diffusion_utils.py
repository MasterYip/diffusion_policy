from ast import Tuple
import math
import torch
from torch import nn
from einops import rearrange, parse_shape

################################################
# Diffusion Beta Schedules
################################################


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """
    Extracts elements from tensor `a` based on indices `t` and reshapes the output tensor.

    Args:
        a (torch.Tensor): Input tensor.
        t (torch.Tensor): Indices tensor.
        x_shape (tuple): Shape of the output tensor.

    Returns:
        torch.Tensor: Extracted tensor with reshaped dimensions.
    """
    # FIXME: in diffusion forcing is (f, b) f = frame, b = batch;
    # in diffusion policy is (B, T, D)
    # There may be a bug here
    if len(t.shape) == 1:
        f = t.shape[0]
        b = 1
    elif len(t.shape) == 2:
        f, b = t.shape  # frames, batch
    elif len(t.shape) == 3:
        f, b, _ = t.shape
    out = a[t]
    # *((1,) * (len(x_shape) - 2)): creates a tuple of ones with a length equal to len(x_shape) - 2.
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class EinopsWrapper(nn.Module):
    def __init__(self, from_shape: str, to_shape: str, module: nn.Module):
        super().__init__()
        self.module = module
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x: torch.Tensor, *args, **kwargs):
        axes_lengths = parse_shape(x, pattern=self.from_shape)
        x = rearrange(x, f"{self.from_shape} -> {self.to_shape}")
        x = self.module(x, *args, **kwargs)
        x = rearrange(x, f"{self.to_shape} -> {self.from_shape}", **axes_lengths)
        return x


def get_einops_wrapped_module(module, from_shape: str, to_shape: str):
    class WrappedModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.wrapper = EinopsWrapper(from_shape, to_shape, module(*args, **kwargs))

        def forward(self, x: torch.Tensor, *args, **kwargs):
            return self.wrapper(x, *args, **kwargs)

    return WrappedModule

################################################
# Noise Level Schedulers
################################################


def exp_noise_mask(horizon, max_noise_level, sigma=2.0, pad_zero=1,
                   dtype=torch.int64):
    """
    noise_level
        ^
    max |                                          *
        |                                       *
        |                                    *
        |                                 *
        |                             *
        |                         *
        |                     *
        |                 *
        |           *
        |      *
    0   |*****_________________________________________→ timestep
        pad_zero                          horizon

    Args:
        horizon (int): Total length of the output tensor
        max_noise_level (int/float): Maximum noise level to reach
        sigma (float, optional): Controls steepness of exponential curve. Defaults to 2.0.
        pad_zero (int, optional): Number of initial timesteps with zero noise. Defaults to 1.
        dtype (torch.dtype, optional): Data type of output tensor. Defaults to torch.int64.

    Returns:
        torch.Tensor: A tensor of shape (horizon,) with the noise mask values
    """

    zeros = torch.zeros(pad_zero)
    len_exp = horizon - pad_zero
    exps = torch.tensor([math.exp((k-len_exp)*sigma / len_exp) for k in range(len_exp)])
    exps = ((exps - math.exp(-sigma)) / (1.0 - math.exp(-sigma)) * max_noise_level)
    return torch.cat([zeros, exps]).to(dtype)


class BaseNoiseScheduler:
    def __init__(self,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        self.dtype = dtype
        self.device = device
        self.from_noise_levels = None
        self.to_noise_levels = None

        self.is_updated = False
        pass

    def reset(self):
        """
        Reset scheduler to initial state
        """
        self.is_updated = False

    def update(self) -> bool:
        """
        Update noise levels

        :return: True as stop signal
        """
        if not self.is_updated:
            self.is_updated = True
            return False
        else:
            return True

    def get_noise_schedule(self, batch_size=1):
        raise NotImplementedError("Must be implemented to return Tuple[from_noise_levels, to_noise_levels]")


class ConstLevelNoiseScheduler(BaseNoiseScheduler):
    def __init__(self,
                 max_level: int,
                 horizon: int,
                 level_subdivision: int = 10,
                 dtype: torch.dtype = torch.int64,
                 device: torch.device = torch.device('cuda:0'),
                 ):

        self.max_level = max_level  # ramge: [0, max_level-1]
        self.horizon = horizon
        self.level_subdivision = level_subdivision
        super().__init__(dtype, device)
        self.progress_cnt = level_subdivision
        self.subdivide_levels = torch.linspace(0, max_level-1, level_subdivision).to(self.device, dtype=dtype)

    # === Interface === #

    def reset(self):
        """
        Reset scheduler to initial state
        """
        self.progress_cnt = self.level_subdivision

    def update(self) -> bool:
        self.progress_cnt -= 1
        if self.progress_cnt <= 0:
            return True
        self.from_noise_levels = (torch.ones(1, self.horizon).to(self.device)
                                  * self.subdivide_levels[self.progress_cnt]).to(self.device, dtype=self.dtype)
        self.to_noise_levels = (torch.ones(1, self.horizon).to(self.device)
                                * self.subdivide_levels[self.progress_cnt - 1]).to(self.device, dtype=self.dtype)
        return False

    def get_noise_schedule(self, batch_size=1):
        return self.from_noise_levels.repeat(batch_size, 1).to(self.device), \
            self.to_noise_levels.repeat(batch_size, 1).to(self.device)


class ShiftBackNoiseScheduler(BaseNoiseScheduler):
    def __init__(self,
                 max_level: int,
                 horizon: int,
                 sigma: float = 2.0,
                 pad_zero: int = 1,
                 dtype: torch.dtype = torch.int64,
                 device: torch.device = torch.device('cuda:0'),
                 ):
        super().__init__(dtype, device)
        self.max_level = max_level
        self.horizon = horizon
        self.sigma = sigma
        self.pad_zero = pad_zero
        self.to_noise_levels = exp_noise_mask(self.horizon, self.max_level, self.sigma,
                                              self.pad_zero, self.dtype).to(self.device)
        self.from_noise_levels = self.shift_noise_mask(self.to_noise_levels)

    def shift_noise_mask(self, noise_mask):
        return torch.cat([noise_mask[1:], noise_mask[-1].unsqueeze(0)], dim=0).to(self.device)

    # === Interface === #
    def get_noise_schedule(self, batch_size=1):
        return self.from_noise_levels.repeat(batch_size, 1), \
            self.to_noise_levels.repeat(batch_size, 1)
