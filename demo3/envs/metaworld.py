import numpy as np
import gymnasium as gym

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from envs.wrappers.vectorized import Vectorized
from envs.tasks.mw_stages import getRewardWrapper

from envs.utils import convert_observation_to_space


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.max_episode_steps = cfg.max_episode_steps

        # Adapt rendering size (this only works for gym <= 0.29)
        self.env.mujoco_renderer.model.vis.global_.offwidth = cfg.camera.image_size
        self.env.mujoco_renderer.model.vis.global_.offheight = cfg.camera.image_size

        self._state_obs, _ = super().reset()
        self.observation_space = convert_observation_to_space(
            self.get_obs(self.cfg.obs)
        )

    def get_obs(self, obs_type=None):
        obs_type = self.cfg.obs if obs_type is None else obs_type
        if obs_type == "state":
            return self._state_obs
        elif obs_type in ("rgbd", "rgb"):
            return {
                "state": self._get_robot_state(),
                "rgb_base": self._get_pixel_obs(),
            }
        else:
            raise NotImplementedError

    def reset(self, **kwargs):
        self._t = 0
        obs, info = super().reset(**kwargs)
        obs = self.env.step(np.zeros(self.env.action_space.shape))[0]
        self._state_obs = obs.astype(np.float32)
        return self.get_obs(self.cfg.obs), info

    def step(self, action):
        reward = 0
        for _ in range(self.cfg.action_repeat):
            obs, r, terminated, _, info = self.env.step(action.copy())
            reward = r  # Options: max, sum, min
        self._state_obs = obs.astype(np.float32)
        self._t += 1
        done = self._t >= self.max_episode_steps
        return self.get_obs(self.cfg.obs), reward, terminated, done, info

    def _get_robot_state(self):
        state = self._state_obs.astype(np.float32)
        return state[0:4]  # Current gripper hand state

    def _get_pixel_obs(self):
        img = self.render()
        return img.transpose(2, 0, 1)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        # BUG: MuJoco Rendering bug, corner images are flipped for some reason
        if self.env.camera_name in ("corner", "corner2"):
            return np.flip(self.env.render(*args, **kwargs), axis=0)
        return self.env.render(*args, **kwargs)

    def get_state(self):
        return self.env.get_env_state()

    def set_state(self, env_state):
        self.env.set_env_state(env_state)


def _make_env(cfg):
    """
    Make Meta-World environment.
    """
    parts = cfg.task.split("-")  # Format is "env-task_id-reward_type"
    env_id = "-".join(parts[1:-1]) + "-v2-goal-observable"
    if (
        not cfg.task.startswith("mw-")
        or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    ):
        raise ValueError("Unknown task:", cfg.task)
    cfg.metaworld.reward_mode = parts[-1]
    cfg.metaworld.obs = cfg.get("obs", "state")
    if cfg.metaworld.reward_mode == "semi":
        cfg.metaworld.reward_mode = "semi_sparse"
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](
        seed=cfg.seed,
        render_mode=cfg.metaworld.render_mode,
    )
    env.camera_name = "corner2"
    env._freeze_rand_vec = False
    env = getRewardWrapper(env_id)(env, cfg.metaworld)
    env = MetaWorldWrapper(env, cfg.metaworld)
    return env


def make_env(cfg):
    """
    Make Vectorized Meta-World environment.
    """
    env = Vectorized(cfg, _make_env)
    cfg.action_penalty = cfg.metaworld.action_penalty
    if isinstance(cfg.max_bc_steps, str):
        cfg.max_bc_steps = cfg.metaworld.max_bc_steps
    return env
