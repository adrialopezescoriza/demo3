import numpy as np
import gymnasium as gym

from collections import OrderedDict
from gymnasium.spaces import Box, Dict


def flatten_space(space):
    high, low = [], []
    obs_shp = []
    for v in space.values():
        try:
            shp = np.prod(v.shape)
        except:
            shp = 1
        obs_shp.append(shp)
        high.append(v.high)
        low.append(v.low)
    obs_shp = (int(np.sum(obs_shp)),)
    return gym.spaces.Box(
        low=np.concatenate(low, axis=-1),
        high=np.concatenate(high, axis=-1),
        dtype=np.float32,
    )


def convert_observation_to_space(observation):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from gymnasium.envs.mujoco_env
    """
    if isinstance(observation, (dict)):
        # if not isinstance(observation, OrderedDict):
        #     warn("observation is not an OrderedDict. Keys are {}".format(observation.keys()))
        space = Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=observation.dtype)
        high = np.full(observation.shape, float("inf"), dtype=observation.dtype)
        space = Box(low, high, dtype=observation.dtype)
    else:
        import torch

        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
            return convert_observation_to_space(observation)
        else:
            raise NotImplementedError(type(observation), observation)

    return space
