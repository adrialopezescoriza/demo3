import numpy as np
import gymnasium as gym


class RobosuiteTask(gym.Env):
    def __init__(
        self,
        env_name,
        cameras=(0, 1),
        reward_type="dense",
        from_pixels=True,
        height=100,
        width=100,
        channels_first=True,
        control=None,
        set_done_at_success=True,
    ):
        self.camera_names = ["frontview", "robot0_eye_in_hand"]

        import robosuite as suite
        from robosuite import load_controller_config

        config = load_controller_config(default_controller="OSC_POSE")
        self.reward_type = reward_type

        if "lift" in env_name:
            env = suite.make(
                env_name="Lift",
                robots="Panda",
                controller_configs=config,
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                control_freq=10,
                horizon=40,
                reward_shaping=True if self.reward_type == "dense" else False,
            )
            self.horizon = 40
        elif "stack" in env_name:
            env = suite.make(
                env_name="Stack",
                robots="Panda",
                controller_configs=config,
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                control_freq=10,
                horizon=80,
                reward_shaping=True if self.reward_type == "dense" else False,
            )
            self.horizon = 80
        elif "door" in env_name:
            # create environment instance
            env = suite.make(
                env_name="Door",
                robots="Panda",
                controller_configs=config,
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                control_freq=10,
                horizon=80,
                reward_shaping=True if self.reward_type == "dense" else False,
            )
            self.horizon = 80
        elif "pick-place-can" in env_name:
            env = suite.make(
                env_name="PickPlace",
                robots="Panda",
                controller_configs=config,
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                control_freq=10,
                horizon=120,
                single_object_mode=2,
                object_type="can",
                reward_shaping=True if self.reward_type == "dense" else False,
            )
            self.horizon = 120

        self._env = env
        self.cameras = cameras
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.channels_first = channels_first
        self.control = control
        self.set_done_at_success = set_done_at_success
        self.domain_name = env_name

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=float)

        if self.from_pixels:
            shape = (
                [3 * len(cameras), height, width]
                if channels_first
                else [height, width, 3 * len(cameras)]
            )
            self._observation_space = gym.spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            shape = (self._unpack_obs(self._env._get_observations()).shape[-1],)
            self._observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.uint8
            )

    def get_observation(self):
        obs = self._env._get_observations()
        return self._unpack_obs(obs)

    def _unpack_obs(self, obs):
        if self.from_pixels:
            images = {}
            for c in self.cameras:
                images[self.camera_names[c]] = obs[self.camera_names[c] + "_image"][
                    ::-1
                ].astype(np.uint8)
                if self.channels_first:
                    images[self.camera_names[c]] = images[
                        self.camera_names[c]
                    ].transpose((2, 0, 1))
            return images
        else:
            robot_state = obs["robot0_proprio-state"]
            object_state = obs["object-state"]
            return np.concatenate((robot_state, object_state), axis=-1)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._unpack_obs(obs)
        info["success"] = self._env._check_success()
        if isinstance(info["success"], np.bool_):
            info["success"] = info["success"].item()
        return obs, reward, False, False, info

    def reward(self):
        return self._env.reward(action=None)

    def reset(self, **kwargs):
        obs = self._env.reset()
        return self._unpack_obs(obs), {"success": self._env._check_success()}

    def render(self, mode="rgb_array", **kwargs):
        obs = self._env._get_observations()
        return obs["frontview_image"][::-1]

    @property
    def _max_episode_steps(self):
        return self.horizon

    @property
    def max_episode_steps(self):
        return self.horizon

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def n_stages(self):
        return 1

    @property
    def reward_mode(self):
        return "dense" if self.reward_type == "dense" else "semi_sparse"
