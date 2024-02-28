import torch
from torch import nn
from torch.distributions import Normal

from models.ppo_agent import ActorCriticAgent
from models.utils import layer_init


class SafeActorCriticAgent(ActorCriticAgent):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size=input_size, output_size=output_size)
        self.classk_size = 1

        # override actor model
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.output_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.output_size))

        self.actor_k_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.classk_size), std=0.01),
        )
        self.actor_k_logstd = nn.Parameter(torch.zeros(1, self.classk_size))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_k=None):
        action_z = self.actor_backbone(x)

        action_mean = self.actor_mean(action_z)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        action_k_mean = self.actor_k_mean(action_z)
        action_k_logstd = self.actor_k_logstd.expand_as(action_k_mean)
        action_k_std = torch.exp(action_k_logstd)
        probs_k = Normal(action_k_mean, action_k_std)

        if action is None:
            action = probs.sample()

        if action_k is None:
            action_k = probs_k.sample()

        log_probs = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        class_k_log_probs = probs_k.log_prob(action_k).sum(1)
        class_k_entropy = probs_k.entropy().sum(1)
        value = self.critic(x)

        results = {
            "action": action,
            "log_prob": log_probs,
            "entropy": entropy,
            "class_k": action_k,
            "class_k_log_prob": class_k_log_probs,
            "class_k_entropy": class_k_entropy,
            "value": value,
        }

        batch_sz = x.shape[0]
        for k, batch in results.items():
            assert batch.shape[0] == batch_sz, f"wrong {k} shape: {batch.shape}"

        return results
