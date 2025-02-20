import random

import numpy as np
import torch

from collections import deque


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeedScheduler:
    def __init__(self, enable=True, sr_thresehold=0.3, num_envs=1):
        self.sr_thresehold = sr_thresehold
        self.enable = enable
        self.num_envs = num_envs
        self._started = False
        self.sr = deque([], maxlen=5)

    def start(self, init_seed=0, max_seeds=1e4, init_counter=1):
        self.max_seeds = int(max_seeds)
        self.seeds = np.arange(
            start=init_seed,
            stop=init_seed + max_seeds,
            step=self.num_envs,
            dtype=np.int32,
        )
        self.counter = init_counter
        self._started = True

    def step(self, sr) -> None:
        if not self._started:
            self.start()
        self.sr.append(sr)
        if (
            self.enable
            and len(self.sr) == self.sr.maxlen
            and np.mean(self.sr) > self.sr_thresehold
            and self.counter < self.max_seeds
        ):
            new_counter = min(self.max_seeds, self.counter * 2)
            print(f"Updated number of seeds from {self.counter} to {new_counter}.")
            self.counter = new_counter
            self.sr.clear()

    def sample(self) -> int:
        if not self._started:
            self.start()
        if self.enable:
            return int(np.random.choice(self.seeds[: self.counter], replace=False))
        return np.random.randint(2**31)
