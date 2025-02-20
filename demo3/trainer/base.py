from common.seed import SeedScheduler


class Trainer:
    """Base trainer class for TD-MPC2."""

    def __init__(self, cfg, env, agent, buffer, logger):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.seed_scheduler = SeedScheduler(
            enable=cfg.seed_scheduler, num_envs=self.cfg.num_envs
        )
        print("Architecture:", self.agent.model)

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        raise NotImplementedError

    def train(self):
        """Train a TD-MPC2 agent."""
        raise NotImplementedError
