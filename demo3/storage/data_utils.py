import numpy as np
import pickle
import torch
import random

from tensordict.tensordict import TensorDict


def load_dataset_as_td(path, num_traj=None, success_only=False):
    """
    Dataset must be stored as list of np.arrays
    Returns list of TensorDicts with different lengths

    trajectories is a list of trajectory
    trajectories[0] has keys like: ['actions', 'dones', ...]

    Note: next_observations means the observation resulting from a given action
    """
    with open(path, "rb") as f:
        trajectories = pickle.load(f)
        random.shuffle(trajectories)
    if success_only:
        trajectories = [t for t in trajectories if t["infos"][-1]["success"]]
    if num_traj is not None:
        trajectories = trajectories[:num_traj]

    def episode_to_tensor(episode):
        if isinstance(episode[0], (torch.Tensor, TensorDict)):
            return torch.stack([o for o in episode])
        elif isinstance(episode[0], dict):
            return torch.stack([TensorDict(o) for o in episode])
        return torch.stack([torch.tensor(o) for o in episode])

    tds = []
    for traj in trajectories:
        tds.append(
            TensorDict(
                dict(
                    obs=episode_to_tensor(
                        traj[
                            (
                                "next_observations"
                                if "next_observations" in traj.keys()
                                else "observations"
                            )
                        ]
                    ),
                    reward=torch.tensor(traj["rewards"]).int(),
                    action=episode_to_tensor(traj["actions"]).float(),
                    stage=(
                        torch.ones(len(traj["rewards"]), dtype=torch.int64)
                        * np.nanmax(traj["rewards"])
                    ).int(),
                ),
                batch_size=(len(traj["rewards"]),),
            )
        )

    return tds
