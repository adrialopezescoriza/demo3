from copy import deepcopy

import gymnasium as gym

from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch


class Vectorized(gym.Wrapper):
    """
    Vectorized environment for TD-MPC2 online training.
    """

    def __init__(self, cfg, env_fn):
        self.cfg = cfg
        self.num_envs = cfg.num_envs

        def make():
            _cfg = deepcopy(cfg)
            _cfg.num_envs = 1
            _cfg.seed = cfg.seed + np.random.randint(1000)
            return env_fn(_cfg)

        self.env = AsyncVectorEnv([make for _ in range(cfg.num_envs)])
        print(f"Created {cfg.num_envs} environments...")
        env = make()

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.max_episode_steps = env.max_episode_steps

        # DEMO3 specific
        if hasattr(env, "n_stages"):
            self.reward_mode = env.reward_mode
            self.n_stages = env.n_stages

    def rand_act(self):
        return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action.numpy())
        if "final_observation" in info.keys():
            if isinstance(obs, dict):
                obs = {
                    k: np.stack([dic[k] for dic in info["final_observation"]], axis=0)
                    for k in info["final_observation"][0]
                }
            else:
                obs = np.stack(info["final_observation"], axis=0)
            info = {
                k: [dic[k] for dic in info["final_info"]] for k in info["final_info"][0]
            }
        return obs, r, terminated, truncated, info

    def render(self, render_all=False):
        return self.env.call("render")[0] if not render_all else self.env.call("render")

    def get_obs(self, *args, **kwargs):
        obs = self.env.call("get_obs", *args, **kwargs)
        if isinstance(obs[0], dict):
            return {k: np.stack([dic[k] for dic in obs], axis=0) for k in obs[0]}
        return np.stack(obs, axis=0)

    def reward(self):
        return self.env.call("reward")

    def get_state(self):
        return self.env.call("get_state")
