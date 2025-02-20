from time import time

import numpy as np
import torch
from termcolor import colored
from tensordict.tensordict import TensorDict
from copy import deepcopy

from trainer.base import Trainer


class ModemTrainer(Trainer):
    """Trainer class for Modem training. Assumes semi-sparse reward environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._step = 0
        self._pretrain_step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self, pretrain=False):
        """Evaluate agent."""
        ep_rewards, ep_max_rewards, ep_successes, ep_seeds = [], [], [], []
        for i in range(max(1, self.cfg.eval_episodes // self.cfg.num_envs)):
            seed = np.random.randint(2**31)
            obs, done, ep_reward, ep_max_reward, t = (
                self.env.reset(seed=seed),
                torch.tensor(False),
                0,
                None,
                0,
            )
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=True)
            while not done.any():
                torch.compiler.cudagraph_mark_step_begin()
                action = (
                    self.agent.policy_action(obs, eval_mode=True)
                    if pretrain
                    else self.agent.act(obs, t0=t == 0, eval_mode=True)
                )
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
            ep_successes.append(info["success"].float().mean())
            ep_seeds.append(seed)

        if self.cfg.save_video:
            if pretrain:
                self.logger.video.save(
                    "pretrain/iteration",
                    self._pretrain_step,
                    key="videos/pretrain_video",
                )
            else:
                self.logger.video.save("eval/step", self._step)

        return dict(
            episode_reward=torch.cat(ep_rewards).mean(),
            episode_max_reward=torch.cat(ep_max_rewards).max(),
            episode_success=torch.stack(ep_successes).mean(),
            best_seed=(
                ep_seeds[torch.argmax(torch.stack(ep_rewards).mean(dim=1)).item()]
                if pretrain
                else None
            ),
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

    def pretrain(self):
        """Pretrains agent policy with demonstration data"""
        demo_buffer = self.buffer._offline_buffer
        n_iterations = self.cfg.pretrain.n_epochs
        self.cfg.pretrain.eval_freq = n_iterations // 25
        start_time = time()
        best_model, best_score = deepcopy(self.agent.bc_model.state_dict()), 0

        print(
            colored(
                f"Policy pretraining: {n_iterations} iterations", "red", attrs=["bold"]
            )
        )

        self.agent.bc_model.train()
        for self._pretrain_step in range(n_iterations):
            metrics = self.agent.init_bc(demo_buffer)

            if self._pretrain_step % self.cfg.pretrain.eval_freq == 0:
                eval_metrics = self.eval(pretrain=True)
                eval_metrics.update({"iteration": self._pretrain_step})
                self.logger.log(eval_metrics, category="pretrain")

                if eval_metrics["episode_reward"] > best_score:
                    best_model = deepcopy(self.agent.bc_model.state_dict())
                    best_score = eval_metrics["episode_reward"]
                    best_seed = eval_metrics["best_seed"]

            if self._pretrain_step % self.cfg.pretrain.log_freq == 0:
                metrics.update(
                    {
                        "iteration": self._pretrain_step,
                        "total_time": time() - start_time,
                    }
                )
                self.logger.log(metrics, category="pretrain")

        if best_score == 0:
            best_model = deepcopy(self.agent.bc_model.state_dict())
            best_seed = eval_metrics["best_seed"]

        self.agent.model.eval()
        self.agent.model.load_state_dict(best_model)
        self.agent.bc_model.load_state_dict(best_model)
        self.seed_scheduler.start(init_seed=best_seed, max_seeds=1e4)

    def train(self):
        """Train agent"""

        # Policy pretraining
        if self.cfg.get("policy_pretraining", False):
            self.pretrain()

        # Start interactive training
        print(colored("\nReplay buffer seeding", "yellow", attrs=["bold"]))
        train_metrics, done, eval_next = {}, torch.tensor(True), False
        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Save Agent periodically
            if self._step % self.cfg.save_freq == 0 and self._step > 0:
                print("Saving agent checkpoint...")
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
                    self._ep_idx = self.buffer.add(tds)
                    train_metrics.update(
                        episode_reward=np.nansum(tds["reward"], axis=0).mean(),
                        episode_max_reward=np.nanmax(tds["reward"], axis=0).max(),
                        episode_success=info["success"].float().nanmean(),
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    self.seed_scheduler.step(train_metrics["episode_success"].item())

                obs = self.env.reset(seed=self.seed_scheduler.sample())
                self._tds = [self.to_td(obs, device="cpu")]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                self._alpha = (
                    max(0, self.cfg.max_bc_steps - self._step) / self.cfg.max_bc_steps
                )
                if np.random.random() < self._alpha and self.cfg.get(
                    "policy_pretraining", False
                ):
                    action = self.agent.policy_action(obs, eval_mode=True)
                else:
                    action = self.agent.act(obs, t0=len(self._tds) == 1)
            elif self.cfg.get("policy_pretraining", False):
                action = self.agent.policy_action(obs, eval_mode=True)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward, device="cpu"))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = max(
                        1, int(self.cfg.seed_steps / self.cfg.steps_per_update)
                    )
                    print(colored("\nTraining MoDem Agent", "green", attrs=["bold"]))
                    print(
                        f"Pretraining agent with {num_updates} update steps on seed data..."
                    )
                else:
                    num_updates = max(
                        1, int(self.cfg.num_envs / self.cfg.steps_per_update)
                    )
                for _ in range(num_updates):
                    agent_train_metrics = self.agent.update(
                        self.buffer, action_penalty=self.cfg.action_penalty
                    )
                train_metrics.update(agent_train_metrics)

            self._step += self.cfg.num_envs

        self.logger.finish(self.agent)
