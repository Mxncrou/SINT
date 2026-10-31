import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


# ------------------------------Z-------------------------
def layer_init(layer, gain):

    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer
# ------------------------------Z-------------------------


def observation_split(observation_dim, n_actions):

    n_enemies = n_actions - 6 # Action space is 6 move + attack*enemy
    n_unit_features = (observation_dim - 2) // (2 * n_enemies)
    return{
        "movement features": 4,
        "n_enemies":         n_enemies,
        "n_allies":          n_enemies - 1, # We assume symmetric teams
        "units features":    n_unit_features,
        "own features":      n_unit_features -2 # Same minus relative position + own position, 
                                                # -4 (relative x and y, in range and distance) 
                                                # + 2 (absolute normalised x and y position) = -2
    }


class SharedTransformer(nn.Module):
    def __init__(self, observation_dim, n_actions, d, n_heads, pwffn):

        super().__init__()
 
        self.observation_layout = observation_split(observation_dim, n_actions)
 
        self.own_embedding     = layer_init(layer=nn.Linear(self.observation_layout["movement features"] + self.observation_layout["own features"], d), gain=1.0)
        self.allies_embedding  = layer_init(layer=nn.Linear(self.observation_layout["units features"], d),                                              gain=1.0)
        self.enemies_embedding = layer_init(layer=nn.Linear(self.observation_layout["units features"], d),                                              gain=1.0)
 
        self.attention_norm_1 = nn.LayerNorm(d)
        self.attention_1      = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.pwffn_1          = nn.Sequential(nn.LayerNorm(d),
                                              layer_init(layer=nn.Linear(d, pwffn), gain=np.sqrt(2)),
                                              nn.ReLU(),
                                              layer_init(layer=nn.Linear(pwffn, d), gain=1.0))
 
        self.attention_norm_2 = nn.LayerNorm(d)
        self.attention_2      = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.pwffn_2          = nn.Sequential(nn.LayerNorm(d),
                                              layer_init(layer=nn.Linear(d, pwffn), gain=np.sqrt(2)),
                                              nn.ReLU(),
                                              layer_init(layer=nn.Linear(pwffn, d), gain=1.0))
 
        self.output_norm        = nn.LayerNorm(d)
        self.actor_move         = layer_init(layer=nn.Linear(d, 6), gain=0.01)
        self.actor_attack       = layer_init(layer=nn.Linear(d, 1), gain=0.01)
        self.critic             = layer_init(layer=nn.Linear(d, 1), gain=1.0)


    def forward(self, o, available_actions):

        batch_shape = o.shape[:-1]
        o = o.reshape(-1, o.shape[-1])
 
        n_move    = self.observation_layout["movement features"]
        n_enemies = self.observation_layout["n_enemies"]
        n_allies  = self.observation_layout["n_allies"]
        n_unit    = self.observation_layout["units features"]

        # Below we get the start of each part of the observation o
        # the observation is as follow:
        # [4 move (can i move?) | n enemies times their features | n allies times their features | own features]
        enemies_start = n_move
        allies_start  = enemies_start + n_enemies * n_unit
        own_start     = allies_start + n_allies * n_unit
 
        move    = o[:, :enemies_start]
        enemies = o[:, enemies_start:allies_start].reshape(-1, n_enemies, n_unit)
        allies  = o[:, allies_start:own_start].reshape(-1, n_allies, n_unit)
        own     = o[:, own_start:]

        me_visible      = torch.ones(o.shape[0], 1, dtype=torch.bool, device=o.device)
        enemies_visible = enemies[..., 1] > 0
        allies_visible  = allies[..., 0] > 0
        mask = ~torch.cat([me_visible, enemies_visible, allies_visible], dim=1)   # (B, 10)
 
        x = torch.cat([self.own_embedding(torch.cat([move, own], dim=-1)).unsqueeze(1),
                       self.enemies_embedding(enemies),
                       self.allies_embedding(allies)], dim=1)
 
        h = self.attention_norm_1(x)
        x = x + self.attention_1(h, h, h, key_padding_mask=mask, need_weights=False)[0]
        x = x + self.pwffn_1(x)
 
        h = self.attention_norm_2(x)
        x = x + self.attention_2(h, h, h, key_padding_mask=mask, need_weights=False)[0]
        x = x + self.pwffn_2(x)
 
        me      = self.output_norm(x[:, 0])
        enemies = self.output_norm(x[:, 1 : 1 + n_enemies])
        
        logits  = torch.cat([self.actor_move(me),
                             self.actor_attack(enemies).squeeze(-1)],
                             dim=-1)
        logits  = logits.reshape(*batch_shape, -1).masked_fill(~available_actions, float("-inf"))
        value   = self.critic(me).squeeze(-1).reshape(batch_shape)
        return Categorical(logits=logits), value

class SharedTransformerMini(SharedTransformer):
    def __init__(self, observation_dim, n_actions):
        super().__init__(observation_dim, n_actions, d=48, n_heads=3, pwffn=120)


class SharedTransformerBase(SharedTransformer):
    def __init__(self, observation_dim, n_actions):
        super().__init__(observation_dim, n_actions, d=64, n_heads=4, pwffn=152)


class SharedTransformerLarge(SharedTransformer):
    def __init__(self, observation_dim, n_actions):
        super().__init__(observation_dim, n_actions, d=96, n_heads=6, pwffn=192)


def lookup_actor_params(model):
    return sum(p.numel() for name, p in model.named_parameters() if not name.startswith("critic"))

