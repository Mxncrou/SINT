import json
import os
from collections import deque
from functools import partial

import numpy as np
import torch
import yaml
from absl import logging
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from src.algorithms.buffer import RolloutBuffer
from src.algorithms.IPPO import IPPO
from src.models.MLPs.SharedMLPs import SharedMLPBase, SharedMLPLarge, SharedMLPMini
from tools.ParallelRunner import ParallelRunner

logging.set_verbosity(logging.WARNING)

MODELS     = {"Shared MLP Mini": SharedMLPMini,
              "Shared MLP Base": SharedMLPBase,
              "Shared MLP Large": SharedMLPLarge}

ALGORITHMS = {"IPPO": IPPO}

with open("settings.json") as s:
    settings = json.load(s)

ALGORITHM = settings["algorithm"]
MODEL     = settings["model"]

with open(os.path.join("scenarios", settings["scene"] + ".yaml")) as s:
    scenario = yaml.safe_load(s)

DEVICE      = settings["device"]

MAKE_ENV    = partial(StarCraftCapabilityEnvWrapper, **scenario["env_args"])
N_ENVS      = settings["training"]["n_envs"]

BUFFER_SIZE = settings["training"]["buffer_size"]
N_UPDATES   = -(-scenario["t_max"] // (BUFFER_SIZE * N_ENVS))


EVAL_EVERY  = scenario["test_interval"]
SAVE_EVERY  = 100_000
EVAL_SIZE   = scenario["test_nepisode"]
PATH        = os.path.join(settings["scene"], MODEL, ALGORITHM)

with open(os.path.join("src", "algorithms", settings["algorithm"] + ".yaml")) as s:
    HPARAMS = yaml.safe_load(s)["Hparams"]


def forward(policy, observation):

    with torch.no_grad():
        dist, values = policy(torch.as_tensor(observation["observations"],      device=DEVICE),
                              torch.as_tensor(observation["available_actions"], device=DEVICE))
    return dist, values.cpu().numpy() * observation["alives"] # Value(dead agent) is meaningless


def train(seed):

    torch.manual_seed(seed)
    
    envs      = ParallelRunner(MAKE_ENV, N_ENVS)
    eval_envs = ParallelRunner(MAKE_ENV, 2) if EVAL_EVERY else None
    
    writer    = SummaryWriter(os.path.join("runs", PATH, f"seed_{seed}"))
    # Our winrate approximation use a wilson interval.
    writer.add_custom_scalars({
    "Evaluation": {
        "Greedy winrate (Wilson 95%)": ["Margin", ["eval/winrate", "eval/lower bound", "eval/upper bound"]],
    }})
    
    try:
        info   = envs.env_info
        obs  = envs.reset()

        policy = MODELS[MODEL](info["obs_shape"], info["n_actions"]).to(DEVICE)
        buffer = RolloutBuffer.build(info, BUFFER_SIZE, N_ENVS)
        agent  = ALGORITHMS[ALGORITHM](policy, buffer, total_iters=N_UPDATES, **HPARAMS)

        rewards, wins, timeouts, ep_reward = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100), np.zeros(N_ENVS)

        pbar = tqdm(total=N_UPDATES * BUFFER_SIZE * N_ENVS, unit="step", unit_scale=True, dynamic_ncols=True)
        for update in range(1, N_UPDATES + 1):

            dist, values = forward(policy, obs)
            while not buffer.full:
                actions = dist.sample()
                step, final, next_obs = envs.step(actions.cpu().numpy())
                buffer.store(observations      = obs["observations"],
                             available_actions = obs["available_actions"],
                             actions           = actions.cpu(), 
                             log_probs         = dist.log_prob(actions).cpu(),
                             values            = values,
                             rewards           = step["reward"], 
                             terminated        = step["terminated"] & ~step["truncated"],
                             truncated         = step["truncated"], 
                             alives            = obs["alives"])

                obs, (dist, values) = next_obs, forward(policy, next_obs)
                buffer.next_values[buffer.ptr - 1] = values
                
                if step["truncated"].any():
                    # This condition happen in case an episode hit the time limit
                    # it gives the next values in that case (which cant be get the usual way as the episode i truncated)
                    buffer.next_values[buffer.ptr - 1][step["truncated"]] = forward(policy, final)[1][step["truncated"]]

                
                ep_reward += step["reward"]
                rewards.extend(ep_reward[step["terminated"]])
                wins.extend(step["won"][step["terminated"]])
                timeouts.extend(step["truncated"][step["terminated"]])
                ep_reward[step["terminated"]] = 0

                pbar.update(N_ENVS)

            steps   = update * BUFFER_SIZE * N_ENVS

            metrics = agent.update() | {"training/episode reward": np.mean(rewards), "training/winrate over last 100 episodes": np.mean(wins), "training/timeout rate": np.mean(timeouts)}

            # ------------------------------Z-------------------------
            if EVAL_EVERY and steps // EVAL_EVERY > (steps - BUFFER_SIZE * N_ENVS) // EVAL_EVERY:
            # ------------------------------Z-------------------------
                metrics |= evaluate(policy, eval_envs, EVAL_SIZE, DEVICE)

            for name, value in metrics.items():
                writer.add_scalar(name, value, steps)
            
            pbar.set_postfix(reward=f"{metrics['training/episode reward']:.2f}", winrate=f"{metrics['training/winrate over last 100 episodes']:.2f}")

            # ------------------------------Z-------------------------
            if steps // SAVE_EVERY > (steps - BUFFER_SIZE * N_ENVS) // SAVE_EVERY or update == N_UPDATES:
            # ------------------------------Z-------------------------
                os.makedirs(os.path.join("checkpoints", PATH), exist_ok=True)
                torch.save(policy.state_dict(), os.path.join("checkpoints", PATH, f"seed_{seed}.pt"))
    
    finally:
        envs.close()
        pbar.close()
        if eval_envs is not None:
            eval_envs.close()
        writer.close()


if __name__ == "__main__":
    for seed in settings["training"]["seeds"]:
        train(seed)