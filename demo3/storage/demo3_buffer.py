import torch
from copy import deepcopy

from storage.buffer import Buffer
from common.logger import timeit
from termcolor import colored


class Demo3Buffer(Buffer):
    """
    Ensemble of an offline dataloader and an online replay buffer.
    """

    def __init__(self, cfg):
        _cfg1, _cfg2 = deepcopy(cfg), deepcopy(cfg)
        _cfg1.batch_size = int(cfg.batch_size)
        _cfg2.batch_size = int(cfg.batch_size)
        super().__init__(_cfg1)

        from .data_utils import load_dataset_as_td

        demo_dataset = load_dataset_as_td(
            _cfg2.demo_path,
            num_traj=_cfg2.n_demos,
            success_only=_cfg2.demo_success_only,
        )
        cfg.n_demos = len(demo_dataset)
        _cfg2.buffer_size = (len(demo_dataset) + int(len(demo_dataset) == 1)) * len(
            demo_dataset[0]
        )  # Offline buffer is not dynamically alocated

        cfg.batch_size = _cfg1.batch_size
        # NOTE: Make sure demonstrations contain same type of rewards as online environment!
        self._offline_buffer = Buffer(_cfg2)
        if len(demo_dataset) == 1:
            self._offline_buffer.add(
                demo_dataset[0].unsqueeze(1)
            )  # TODO: This is a patch to solve bug in Replay Buffer when only having 1 trajectory (we duplicate first demo)
        for _td in demo_dataset:
            self._offline_buffer.add(_td.unsqueeze(1))
        print(
            colored(
                f"Filled demo buffer with {self._offline_buffer.num_eps} trajectories",
                "green",
            )
        )

    def sample(self, return_td=False):
        return super().sample()

    # TODO: Need to revisit this to ensure some kind of diversity
    def sample_for_disc(self, batch_size: int):
        if self._offline_buffer.batch_size > 0:
            td0 = self._offline_buffer.sample_single(return_td=True)
            td1 = super().sample_single(return_td=True)

            # Concat + Shuffle
            tds = torch.cat([td0, td1], dim=0)
        else:
            tds = super().sample_single(return_td=True)

        obs_return = []
        for stage_index in range(self.cfg.n_stages):
            # Success indices
            success_indices = (
                (tds["reward"] == stage_index) * (tds["stage"] > stage_index)
            ).nonzero(as_tuple=True)[0]
            fail_indices = (
                (tds["reward"] == stage_index) * (tds["stage"] <= stage_index)
            ).nonzero(as_tuple=True)[0]

            # Cut at minimum to avoid data imbalance
            success_indices = success_indices[
                : min(len(success_indices), len(fail_indices))
            ]
            fail_indices = fail_indices[: min(len(success_indices), len(fail_indices))]

            # Extract observations for those indices
            obs_return.append(
                {
                    "success_data": tds["obs"][success_indices],
                    "fail_data": tds["obs"][fail_indices],
                }
            )

        return obs_return
