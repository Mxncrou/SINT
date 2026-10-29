import os
import time

os.environ["SDL_VIDEODRIVER"] = "x11"  # pygame's Wayland backend crashes on Hyprland

import json

import numpy as np
import torch
import yaml
from absl import logging
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from src.models.MLPs.SharedMLPs import (
    SharedMLPBase,
    SharedMLPLarge,
    SharedMLPMini,
)
from tools.human_render import HumanRender

logging.set_verbosity(logging.WARNING)

MODELS = {"Shared MLP Mini": SharedMLPMini,
          "Shared MLP Base": SharedMLPBase,
          "Shared MLP Large": SharedMLPLarge}

with open("settings.json") as s:
    settings = json.load(s)
with open(os.path.join("scenarios", settings["scene"] + ".yaml")) as s:
    scenario = yaml.safe_load(s)

DEVICE      = settings["device"]
RENDER_MODE = settings["render"]
FPS         = 60   # For 3D
SPS         = 2.8  # For 2D
RUN_PATH    = os.path.join(settings["scene"], settings["model"], settings["algorithm"])


def watch(seed):

    env = StarCraftCapabilityEnvWrapper(**scenario["env_args"])
    
    if RENDER_MODE == "3D":
        env = HumanRender(env, fps=FPS)  # must wrap before the first reset()
    
    info = env.get_env_info()

    policy = MODELS[settings["model"]](info["obs_shape"], info["n_actions"]).to(DEVICE)
    policy.load_state_dict(torch.load(os.path.join("checkpoints", RUN_PATH, f"seed_{seed}.pt"), map_location=DEVICE))

    try:
        while True:
            env.reset()
            
            terminated = False
            while not terminated:
                with torch.no_grad():
                    dist, _ = policy(torch.as_tensor(np.array(env.get_obs()), device=DEVICE),
                                     torch.as_tensor(np.array(env.get_avail_actions(), dtype=bool), device=DEVICE))
                _, terminated, info = env.step(dist.probs.argmax(-1).cpu().numpy())  # greedy

                if RENDER_MODE == "2D":
                    env.render()
                    time.sleep(1 / SPS)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    watch(7)