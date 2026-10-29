import numpy as np
import torch


def wilson(p, n, z=1.96):
 
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half   = z / (1 + z**2 / n) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return center - half, center + half


@torch.no_grad()
def evaluate(policy, envs, n_episodes, device):

    quota  = -(-n_episodes // envs.n_envs)
    played = np.zeros(envs.n_envs, dtype=int)
    wins   = []
    obs    = envs.reset()
    while (played < quota).any():
        dist, _ = policy(torch.as_tensor(obs["observations"],      device=device),
                         torch.as_tensor(obs["available_actions"], device=device))
        step, _, obs = envs.step(dist.probs.argmax(-1).cpu().numpy()) # greedy
        for e in np.flatnonzero(step["terminated"]):
            if played[e] < quota:
                wins.append(float(step["won"][e]))
                played[e] += 1

    p = float(np.mean(wins))
    low, high = wilson(p, len(wins))
    return {"winrate": p, "winrate lower bound": low, "winrate upper bound": high}