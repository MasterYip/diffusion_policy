import numpy as np
import torch
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.obsavoid.obsavoid_env import sine_bound_env, increase_bound_env, randpath_bound_env

class ObsAvoidRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            n_obs_steps,
            n_action_steps,
            max_steps,
        ):
        super().__init__(output_dir)

        # self.env = sine_bound_env(True, y=0, v=0, env_step=0.01)
        # self.env = increase_bound_env(True, y=0, v=0, env_step=0.01)
        # self.env = randpath_bound_env(True, y=0, v=0, env_step=0.01)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps

    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        env = randpath_bound_env(True, y=0, v=0, env_step=0.01)
        obs = env.get_observation()
        obs_hist = [obs for _ in range(self.n_obs_steps)]
        ema_reward = 0
        ema_coeff = 0.99
        for _ in range(self.max_steps):
            obs = env.get_observation()
            obs_hist = obs_hist[1:] + [obs]
            np_obs_array = np.array(obs_hist, dtype=np.float32)
            
            # device transfer
            obs_array = torch.from_numpy(np_obs_array).to(device=device)
            obs_array = obs_array.unsqueeze(0)
            obs_dict = {
                'obs': obs_array,
            }
            
            # predict action
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            
            # device transfer
            np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
            
            # env.step_env(acc=env.get_action()[0])
            env.step_env(acc=np_action_dict["action"][0,0,0])
            ema_reward = env.get_reward() * (1-ema_coeff) + ema_reward * ema_coeff
            
            # # print obs, precision .3f
            # formatted_numbers = ["{:.3f}".format(num) for num in obs]
            # print(formatted_numbers + [str(env.get_action()[0]), str(env.get_reward())])
        env.end()
          
        return {"test_mean_score": ema_reward}
