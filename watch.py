import os
import time

import numpy as np
import torch
import yaml
from absl import logging
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

# ------------------------------Z-------------------------
from pysc2.lib import remote_controller as rc
_step = rc.RemoteController.step
rc.RemoteController.step = lambda self, count=1: [(_step(self, 1), time.sleep(1 / 22.4)) for _ in range(count)]
# ------------------------------Z-------------------------

from src.models.MLPs.SeparatedMLPs import (
    SeparatedMLPBase,
    SeparatedMLPLarge,
    SeparatedMLPMini,
)
from src.models.MLPs.SharedMLPs import (
    SharedMLPBase,
    SharedMLPLarge,
    SharedMLPMini,
)
from src.models.Transformers.SharedTransformers import (
    SharedTransformerBase,
    SharedTransformerLarge,
    SharedTransformerMini,
)

logging.set_verbosity(logging.WARNING)

MODELS = {"Shared MLP Mini"         : SharedMLPMini,
          "Shared MLP Base"         : SharedMLPBase,
          "Shared MLP Large"        : SharedMLPLarge,
          "Separated MLP Mini"      : SeparatedMLPMini,
          "Separated MLP Base"      : SeparatedMLPBase,
          "Separated MLP Large"     : SeparatedMLPLarge,
          "Shared Transformer Mini" : SharedTransformerMini,
          "Shared Transformer Base" : SharedTransformerBase,
          "Shared Transformer Large": SharedTransformerLarge}

with open("config.yaml") as s:
    settings = yaml.safe_load(s)
with open(os.path.join("scenarios", settings["scene"] + ".yaml")) as s:
    scenario = yaml.safe_load(s)

DEVICE      = settings["device"]
RUN_PATH    = os.path.join(settings["scene"], settings["agent"]["model"], settings["agent"]["algorithm"])
HPARAM_PATH = "d68d39a6" # Manual, I don't know how to do it dynamically.

def watch(seed):

    env = StarCraftCapabilityEnvWrapper(**scenario["env_args"])    
    info = env.get_env_info()

    policy = MODELS[settings["agent"]["model"]](info["obs_shape"], info["n_actions"]).to(DEVICE)
    policy.load_state_dict(torch.load(os.path.join("checkpoints", RUN_PATH, HPARAM_PATH, f"seed_{seed}.pt"), map_location=DEVICE))

    try:
        while True:
            env.reset()

            terminated = False
            while not terminated:
                with torch.no_grad():
                    dist, _ = policy(torch.as_tensor(np.array(env.get_obs()), device=DEVICE),
                                     torch.as_tensor(np.array(env.get_avail_actions(), dtype=bool), device=DEVICE))
                _, terminated, info = env.step(dist.probs.argmax(-1).cpu().numpy())  # greedy
                
                time.sleep(1/22.15)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    watch(42)