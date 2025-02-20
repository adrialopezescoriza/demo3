from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate agent."""
        ep_rewards, ep_max_rewards = [], []
        for i in range(max(1, self.cfg.eval_episodes // self.cfg.num_envs)):
            obs, done, ep_reward, ep_max_reward, t = (
                self.env.reset(),
                torch.tensor(False),
                0,
                None,
                0,
            )
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=True)
            while not done.any():
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_max_reward = (
                    torch.maximum(ep_max_reward, reward)
                    if ep_max_reward is not None
                    else reward
                )
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            assert (
                done.all()
            ), "Vectorized environments must reset all environments at once."
            ep_rewards.append(ep_reward)
            ep_max_rewards.append(ep_max_reward)

        if self.cfg.save_video:
            self.logger.video.save("eval/step", self._step)

        return dict(
            episode_reward=torch.cat(ep_rewards).mean(),
            episode_max_reward=torch.cat(ep_max_rewards).max(),
            episode_success=info["success"].float().mean(),
        )

    def to_td(self, obs, action=None, reward=None, device="cpu"):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=())
        else:
            obs = obs.unsqueeze(0)
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan")).repeat(self.cfg.num_envs)
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(
                1,
                self.cfg.num_envs,
            ),
        )
        return td.to(torch.device(device))

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, torch.tensor(True), False
        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Save agent periodically
            if self._step % self.cfg.save_freq == 0 and self._step > 0:
                print("Saving agent and discriminator checkpoints...")
                self.logger.save_agent(self.agent, identifier=f"agent_{self._step}")

            # Reset environment
            if done.any():
                assert (
                    done.all()
                ), "Vectorized environments must reset all environments at once."
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    tds = torch.cat(self._tds)
                    train_metrics.update(
                        episode_reward=np.nansum(tds["reward"], axis=0).mean(),
                        episode_max_reward=np.nanmax(tds["reward"], axis=0).max(),
                        episode_success=info["success"].float().nanmean(),
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    self._ep_idx = self.buffer.add(tds)

                    self.seed_scheduler.step(train_metrics["episode_success"].item())
                obs = self.env.reset(seed=self.seed_scheduler.sample())
                self._tds = [self.to_td(obs, device="cpu")]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward, device="cpu"))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = max(
                        1, int(self.cfg.num_envs / self.cfg.steps_per_update)
                    )
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(
                        self.buffer, action_penalty=self.cfg.action_penalty
                    )
                train_metrics.update(_train_metrics)

            self._step += self.cfg.num_envs

        self.logger.finish(self.agent)
