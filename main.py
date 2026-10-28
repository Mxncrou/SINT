import json
import os
import time
from functools import partial

os.environ["SDL_VIDEODRIVER"] = "x11"

import numpy as np
import torch
import yaml
from absl import logging
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.buffer import RolloutBuffer
from src.algorithms.PPOs.IPPO import IPPO
from src.models.MLPs.SharedMLPs import (
    SharedMLPBase,
    SharedMLPLarge,
    SharedMLPMini,
    lookup_actor_params,
    lookup_params,
)
from tools.human_render import HumanRender
from tools.ParallelRunner import ParallelRunner

# logging.set_verbosity(logging.DEBUG)
logging.set_verbosity(logging.WARNING)

with open("settings.json") as s:
    settings = json.load(s) 

SCENE = settings["scene"]

with open(os.path.join("scenarios/", SCENE + ".yaml")) as s:
    scenario = yaml.safe_load(s)

DEVICE  = settings["device"]

ENV_ARGS = scenario["env_args"]
MODELS  = {
    "Shared MLP Mini"  : SharedMLPMini,
    "Shared MLP Base"  : SharedMLPBase,
    "Shared MLP Large" : SharedMLPLarge
}
MODEL = settings["model"]
MODEL_PARAMS_FILE = settings["algorithm"] + ".pt"

PATH   = os.path.join("checkpoints", SCENE, MODEL)

# For training
N_ENVS        = settings["train"]["n_envs"]
BUFFER_SIZE   = settings["train"]["buffer_size"]
TRAINING_SIZE = scenario["t_max"]

# Evaluation and watching setup
EVALUATION_LENGTH = scenario["test_nepisode"]
RENDER_MODE       = settings["render"]
FPS               = 60  # For 3D
SPS               = 2.8 # For 2D

def train():

    envs = ParallelRunner(partial(StarCraftCapabilityEnvWrapper, **ENV_ARGS), N_ENVS)
    env_info = envs.env_info

    policy = MODELS[MODEL](env_info["obs_shape"], env_info["n_actions"]).to(DEVICE)
    buffer = RolloutBuffer.build(env_info, BUFFER_SIZE, N_ENVS)
    agent  = IPPO(policy, buffer, 
                  learning_rate=5e-4,
                  gamma=0.99,
                  lam=0.95,
                  epsilon=0.2,
                  c_1=0.5,
                  c_2=0.01,
                  epochs=5,
                  n_minibatches=1, 
                # Please, make sure that TRAINING_SIZE(BUFFER_SIZE * N_ENVS) is an integer;
                # if not, it will be rounded, it is not much of a problem in practice tho.
                # - ( - a // b) ceil the result, if you want to floor simply use a // b
                # You could also have problem with number of steps trained,
                # but it shouldnt be that bad as most models converge early.
                  total_iters=-(-TRAINING_SIZE//(BUFFER_SIZE*N_ENVS))) 
    writer = SummaryWriter(os.path.join("runs", SCENE, MODEL, settings["algorithm"]), purge_step=0) 

    observation_dictionary = envs.reset()

    # ------------------------------Z-------------------------
    episode_reward = np.zeros(N_ENVS)
    episode_rewards = []
    wins = []
    # ------------------------------Z-------------------------

    # Envs truncated at the previous step: their next_values were already set from the final observation
    truncated = np.zeros(N_ENVS, dtype=bool)

    pbar = tqdm(range(0, TRAINING_SIZE + BUFFER_SIZE * N_ENVS, N_ENVS), unit="step", unit_scale=N_ENVS, dynamic_ncols=True)
    for steps in pbar:

        with torch.no_grad():
            dist, values = policy(torch.as_tensor(observation_dictionary["observations"],      device=DEVICE),
                                  torch.as_tensor(observation_dictionary["available_actions"], device=DEVICE))
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            next_values = values.cpu().numpy() * observation_dictionary["alives"]
            buffer.next_values[buffer.ptr - 1][~truncated] = next_values[~truncated]
    
        if buffer.full:
            metrics = agent.update()
            mer = np.mean(episode_rewards[-100:])
            winrate = np.mean(wins[-100:])     
            for mn, mv in metrics.items():
                writer.add_scalar(f"{mn}", mv, steps)
            writer.add_scalar("episode reward", mer, steps)
            writer.add_scalar("winrate", winrate, steps)

            pbar.set_postfix(episode_reward=f"{mer:.2f}",
                             winrate=winrate,
                             loss=metrics["loss"])

        
        step, final_observation, next_observation = envs.step(actions.cpu().numpy())

        # ------------------------------Z-------------------------
        episode_reward += step["reward"]
        episode_rewards.extend(episode_reward[step["terminated"]])
        episode_reward[step["terminated"]] = 0
        wins.extend(step["won"][step["terminated"]])
        # ------------------------------Z-------------------------
        
        buffer.store(
            observations=observation_dictionary["observations"],
            available_actions=observation_dictionary["available_actions"],
            actions=actions.cpu(),
            log_probs=log_probs.cpu(),
            values=values.cpu().numpy() * observation_dictionary["alives"],
            rewards=step["reward"],
            terminated=step["terminated"] & ~step["truncated"],
            truncated=step["truncated"],
            alives=observation_dictionary["alives"]
            )

        # ------------------------------Z-------------------------
        truncated = step["truncated"]
        if truncated.any():
            with torch.no_grad():
                _, final_values = policy(torch.as_tensor(final_observation["observations"],      device=DEVICE),
                                         torch.as_tensor(final_observation["available_actions"], device=DEVICE))
            final_values = final_values.cpu().numpy() * final_observation["alives"]
            buffer.next_values[buffer.ptr - 1][truncated] = final_values[truncated]

        observation_dictionary = next_observation
        # ------------------------------Z-------------------------

    os.makedirs(PATH, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(PATH, MODEL_PARAMS_FILE))
    envs.close()

def evaluate():

    env = StarCraftCapabilityEnvWrapper(**ENV_ARGS)
    env_info = env.get_env_info()

    policy = MODELS[MODEL](env_info["obs_shape"], env_info["n_actions"]).to(DEVICE)
    policy.load_state_dict(torch.load(os.path.join(PATH, MODEL_PARAMS_FILE), map_location=DEVICE))

    wins = []
    for e in range(EVALUATION_LENGTH):
        env.reset()
        terminated = False

        while not terminated:
            obs = torch.as_tensor(np.array(env.get_obs()), device=DEVICE)
            available_actions = torch.as_tensor(np.array(env.get_avail_actions(), dtype=bool), device=DEVICE)

            with torch.no_grad():
                dist, _ = policy(obs, available_actions)
            actions = dist.probs.argmax(-1).cpu().numpy()

            _, terminated, info = env.step(actions)

        wins.append(info.get("battle_won", False))
        winrate = np.mean(wins)

    print(f"Win rate: {winrate * 100:.2f}%", f" ±: {1.96 * np.sqrt((winrate*(1-winrate))/EVALUATION_LENGTH) * 100:.2f}%")
    env.close()

def watch():

    env = StarCraftCapabilityEnvWrapper(**ENV_ARGS)
    if RENDER_MODE == "3D":
        env = HumanRender(env, fps=FPS)
    env_info = env.get_env_info()

    policy = MODELS[MODEL](env_info["obs_shape"], env_info["n_actions"]).to(DEVICE)
    policy.load_state_dict(torch.load(os.path.join(PATH, MODEL_PARAMS_FILE), map_location=DEVICE))

    try:
        while True:
            env.reset()
            terminated = False

            while not terminated:
                obs = torch.as_tensor(np.array(env.get_obs()), device=DEVICE)
                available_actions = torch.as_tensor(np.array(env.get_avail_actions(), dtype=bool), device=DEVICE)

                with torch.no_grad():
                    dist, _ = policy(obs, available_actions)
                actions = dist.probs.argmax(-1).cpu().numpy()

                _, terminated, _ = env.step(actions)
                if RENDER_MODE == "2D":
                    env.render()
                    time.sleep(1 / SPS)

    except KeyboardInterrupt:
        pass

    finally:
        env.close()
        print("Actor Parameters", lookup_actor_params(policy))
        print("Total Parameters", lookup_params(policy))


if __name__ == "__main__":
    train()
