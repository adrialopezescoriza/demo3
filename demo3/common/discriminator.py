import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict.tensordict import TensorDict


class Discriminator(nn.Module):
    def __init__(self, envs, cfg, state_shape=None, compile=False):
        super().__init__()
        self.n_stages = envs.n_stages
        state_shape = (
            np.prod(state_shape)
            if state_shape
            else np.prod(envs.observation_space.shape)
        )
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_shape, 32),
                    nn.Sigmoid(),
                    nn.Linear(32, 1),
                )
                for _ in range(self.n_stages)
            ]
        )
        self.trained = [False] * self.n_stages
        self._cfg = cfg
        self.device = torch.device("cuda:0")
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=cfg.disc_lr, capturable=True)

        if compile:
            print("compiling - discriminator update")
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
                fp (str): Filepath to save state dict to.
        """
        torch.save({"discriminator": self.state_dict()}, fp)

    def set_trained(self, stage_idx):
        self.trained[stage_idx] = True

    def forward(self, next_s, stage_idx):
        net = self.nets[stage_idx]
        return net(next_s)

    def update(self, buffer, **kwargs):
        data = buffer.sample_for_disc(self._cfg.batch_size)
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(data, **kwargs)

    def _update(self, data, encoder_function=None):
        disc_losses = []
        for stage_idx in range(self.n_stages):
            success_data = data[stage_idx]["success_data"]
            if len(success_data) == 0:
                continue
            fail_data = data[stage_idx]["fail_data"]

            success_data = success_data[
                torch.randperm(success_data.size(0))[: self._cfg.batch_size]
            ]  # shuffle and cut
            fail_data = fail_data[
                torch.randperm(fail_data.size(0))[: self._cfg.batch_size]
            ]  # shuffle and cut

            disc_next_obs = torch.cat([fail_data, success_data], dim=0)
            disc_labels = torch.cat(
                [
                    torch.zeros(
                        (len(fail_data), 1), device=self.device
                    ),  # fail label is 0
                    torch.ones(
                        (len(success_data), 1), device=self.device
                    ),  # success label is 1
                ],
                dim=0,
            )

            if encoder_function:
                with torch.no_grad():
                    disc_next_obs = encoder_function(disc_next_obs)

            logits = self(disc_next_obs, stage_idx)
            disc_loss = F.binary_cross_entropy_with_logits(logits, disc_labels)

            self.optimizer.zero_grad()
            disc_loss.backward()
            self.optimizer.step()

            disc_losses += [float(disc_loss.mean().item())]

        return TensorDict(
            {"discriminator_loss": np.mean(disc_losses)}
            if len(disc_losses) != 0
            else {}
        )

    def get_reward(self, next_s, stage_idx):
        """
        next_s : torch.Tensor (length, batch_size, state_dim)
        stage_idx : torch.Tensor (length, batch_size, 1)
        """
        with torch.no_grad():
            assert (stage_idx >= 0).all() and (
                stage_idx <= self.n_stages
            ).all(), "Stage idx is out of bounds!"
            bs = stage_idx.shape[:-1]

            stage_rewards = [
                (
                    torch.tanh(self(next_s, stage_idx=i))
                    if self.trained[i]
                    else torch.zeros(*bs + (1,), device=next_s.device)
                )
                for i in range(self.n_stages)
            ]
            stage_rewards = torch.cat(
                stage_rewards + [torch.zeros(*bs + (1,), device=next_s.device)], dim=-1
            )  # Dummy stage_reward for success state (always 0)

            k = 3
            reward = k * stage_idx + torch.gather(
                stage_rewards, -1, stage_idx.long()
            )  # Selects stage index for each reward
            reward = reward / (k * self.n_stages)  # Normalize reward (approx.)
            reward = reward + 1  # make the reward positive

            return reward
