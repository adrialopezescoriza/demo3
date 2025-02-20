from collections import defaultdict
from tensordict.tensordict import TensorDict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    """

    def __init__(self, env):
        super().__init__(env)
        self.max_episode_steps = env.max_episode_steps
        self.num_envs = env.num_envs if hasattr(env, "num_envs") else 1
        self._wrapped_vectorized = hasattr(env, "num_envs")

    def rand_act(self):
        return self._try_f32_tensor(self.env.rand_act())

    def _try_f32_tensor(self, x):
        x = torch.from_numpy(x.copy()) if isinstance(x, np.ndarray) else x
        if x.dtype == torch.float64:
            x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
            obs = TensorDict(
                obs, batch_size=self.num_envs if self._wrapped_vectorized else ()
            )
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, task_idx=None, **kwargs):
        # WARNING: We lose the reset info in this last wrapper
        obs, _ = self.env.reset(**kwargs)
        return self._obs_to_tensor(obs)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def step(self, action):
        obs, reward, _, done, info = self.env.step(action)
        obs = self._obs_to_tensor(obs)
        if isinstance(info, tuple):
            info = {
                key: torch.stack([torch.tensor(d[key]) for d in info])
                for key in info[0].keys()
            }
            if "success" not in info.keys():
                info["success"] = torch.zeros(len(done))
        else:
            info = defaultdict(float, info)
            info = TensorDict(info)
        return obs, torch.tensor(reward, dtype=torch.float32), torch.tensor(done), info

    def get_obs(self, *args, **kwargs):
        return self._obs_to_tensor(self.env.get_obs(*args, **kwargs))

    def reward(self, *args, **kwargs):
        return torch.tensor(self.env.reward(*args, **kwargs), dtype=torch.float32)
