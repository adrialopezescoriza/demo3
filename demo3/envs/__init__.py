from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.tensor import TensorWrapper


def missing_dependencies(task):
    raise ValueError(
        f"Missing dependencies for task {task}; install dependencies to use this environment."
    )


try:
    from envs.maniskill import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies
try:
    from envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from envs.robosuite import make_env as make_robosuite_env
except:
    make_robosuite_env = missing_dependencies


warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_env(cfg):
    """
    Make a vectorized environment for TD-MPC2 experiments.
    """
    gym.logger.set_level(40)

    env = None
    for fn in [make_maniskill_env, make_metaworld_env, make_robosuite_env]:
        try:
            env = fn(cfg)
            break
        except ValueError:
            pass
    if env is None:
        raise ValueError(
            f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.'
        )
    env = TensorWrapper(env)
    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:  # Box
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length) * cfg.num_envs
    return env
