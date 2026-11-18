import numpy as np
from torch import nn
from torch.distributions import Categorical


# ------------------------------Z-------------------------
def layer_init(layer, gain):

    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer
# ------------------------------Z-------------------------


class SeparatedMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, d):

        super().__init__()

        self.actor = nn.Sequential(
            layer_init(layer=nn.Linear(obs_dim, d), gain=np.sqrt(2)),
            nn.ReLU(),
            layer_init(layer=nn.Linear(d, d), gain=np.sqrt(2)),
            nn.ReLU(),
            layer_init(layer=nn.Linear(d, d), gain=np.sqrt(2)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(d, act_dim), gain=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(layer=nn.Linear(obs_dim, d), gain=np.sqrt(2)),
            nn.ReLU(),
            layer_init(layer=nn.Linear(d, d), gain=np.sqrt(2)),
            nn.ReLU(),
            layer_init(layer=nn.Linear(d, d), gain=np.sqrt(2)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(d, 1), gain=1.0)
        )

    def forward(self, o, available_actions):

        logits = self.actor(o).masked_fill(~available_actions, float("-inf"))
        value  = self.critic(o).squeeze(-1)
        return Categorical(logits=logits), value


class SeparatedMLPMini(SeparatedMLP):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim, d=128)


class SeparatedMLPBase(SeparatedMLP):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim, d=170)


class SeparatedMLPLarge(SeparatedMLP):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim, d=256)


def lookup_params(model):
    return sum(p.numel() for p in model.parameters())


def lookup_actor_params(model):
    return lookup_params(model.actor) + lookup_params(model.shared)