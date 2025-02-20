import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"
import warnings

warnings.filterwarnings("ignore")
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from storage.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from common.logger import Logger
from trainer.online_trainer import OnlineTrainer
from trainer.demo3_trainer import Demo3Trainer
from trainer.modem_trainer import ModemTrainer
from storage.ensemble_buffer import EnsembleBuffer
from storage.demo3_buffer import Demo3Buffer

torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")


@hydra.main(config_name="demo3", config_path="./config/")
def train(cfg: dict):
    """
    Script for training single-task / multi-task TD-MPC2 agents.

    Most relevant args:
            `task`: task name (or mt30/mt80 for multi-task training)
            `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
            `steps`: number of training/environment steps (default: 10M)
            `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
            $ python train.py task=mt80 model_size=48
            $ python train.py task=mt30 model_size=317
            $ python train.py task=dog-run steps=7000000
    ```
    """
    # Config checks and processing
    assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg.demo_sampling_ratio = cfg.get("demo_sampling_ratio", 0.0)
    cfg.use_demos = cfg.demo_sampling_ratio > 0.0
    assert (
        cfg.demo_sampling_ratio >= 0.0 and cfg.demo_sampling_ratio <= 1.0
    ), f"Oversampling ratio {cfg.demo_sampling_ratio} is not between 0 and 1"
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    # Initiallize elements
    env_ = make_env(cfg)
    if cfg.enable_reward_learning:
        # DEMO3
        cfg.algorithm = "DEMO3" if cfg.use_demos else "TD-MPC2 + Reward Learning"
        trainer_cls = Demo3Trainer
        cfg.n_stages = env_.n_stages
        buffer_cls = EnsembleBuffer if cfg.use_demos else Demo3Buffer
    elif cfg.use_demos:
        # MoDem
        cfg.algorithm = "Modem-v2"
        trainer_cls = ModemTrainer
        buffer_cls = EnsembleBuffer
    else:
        # TDMPC
        cfg.algorithm = "TDMPC2"
        trainer_cls = OnlineTrainer
        buffer_cls = Buffer

    buffer_ = buffer_cls(cfg)
    logger_ = Logger(cfg)

    # Training code
    trainer = trainer_cls(
        cfg=cfg,
        env=env_,
        agent=TDMPC2(cfg),
        buffer=buffer_,
        logger=logger_,
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
