import numpy as np
import gymnasium as gym
import torch
import robosuite as suite

from envs.wrappers.vectorized import Vectorized
from envs.tasks.robosuite_stages import RobosuiteTask
from gymnasium.wrappers.rescale_action import RescaleAction

from envs.utils import convert_observation_to_space


class RobosuiteWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.observation_space = convert_observation_to_space(
            self.select_obs(self.get_observation())
        )

    def select_obs(self, obs):
        if self.env.from_pixels:
            processed = {}
            for k, v in obs.items():
                processed["rgb_" + k] = v
            return processed
        else:
            return obs

    def rand_act(self):
        return self.action_space.sample().astype(np.float32)

    def reset(self, **kwargs):
        self._t = 0
        obs, info = super().reset(**kwargs)
        return self.select_obs(obs), info

    def step(self, action):
        reward = 0
        action = action.numpy() if isinstance(action, torch.Tensor) else action
        for _ in range(self.cfg.action_repeat):
            obs, r, terminated, _, info = self.env.step(action)
            reward = r  # Options: max, sum, min
        self._t += 1
        done = self._t >= self.max_episode_steps
        return self.select_obs(obs), reward, False, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def get_obs(self, *args, **kwargs):
        return self.select_obs(self.get_observation())


def _make_env(cfg):
    """
    Make Robosuite environment.
    """
    parts = cfg.task.split("-")  # Format is "env-task_id-reward_type"
    env_id = "-".join(parts[1:-1])
    reward_mode = parts[-1]
    if not cfg.task.startswith("robosuite-"):
        raise ValueError("Unknown task:", cfg.task)
    env = RobosuiteTask(
        env_name=env_id,
        from_pixels=cfg.obs in ("rgb", "rgbd"),
        reward_type=reward_mode,
        cameras=(0, 1),
        height=cfg.robosuite.camera.image_size,
        width=cfg.robosuite.camera.image_size,
        channels_first=True,
        control=None,
        set_done_at_success=False,
    )
    cfg.robosuite.obs = cfg.obs
    env = RescaleAction(env, -1.0, 1.0)
    env = RobosuiteWrapper(env, cfg.robosuite)
    return env


def make_env(cfg):
    """
    Make Vectorized Robosuite environment.
    """
    env = Vectorized(cfg, _make_env)
    cfg.action_penalty = cfg.robosuite.get("action_penalty", False)
    if isinstance(cfg.max_bc_steps, str):
        cfg.max_bc_steps = cfg.robosuite.max_bc_steps
    return env
