import numpy as np


class RolloutBuffer:


    def __init__(self, n_steps, n_agents, n_envs, obs_shape, n_actions):

        T, N, E = n_steps, n_agents, n_envs


        # Agent input relative data
        self.observations      = np.zeros((T, E, N, obs_shape), np.float32)
        self.available_actions = np.zeros((T, E, N, n_actions), np.bool)

        # Agent output relative data
        self.actions           = np.zeros((T, E, N),            np.int64)
        self.log_probs         = np.zeros((T, E, N),            np.float32)
        self.values            = np.zeros((T, E, N),            np.float32)
        self.next_values       = np.zeros((T, E, N),            np.float32)

        # Environment-wide data
        self.rewards           = np.zeros((T, E),               np.float32)
        self.terminated        = np.zeros((T, E),               np.bool)
        self.truncated         = np.zeros((T, E),               np.bool)

        self.alives            = np.zeros((T, E, N),            np.bool)

        # internal variables
        self.n_steps = T
        self.ptr     = 0

    # ------------------------------Z-------------------------
    @property
    def full(self):
        return self.ptr == self.n_steps

    @classmethod
    def build(cls, env_info, n_steps, n_envs=1):

        # env_info return a dict with useful info based on the Wrapper, it is provided by SMACv2
        return cls(
            n_steps   = n_steps,
            n_agents  = env_info["n_agents"],
            n_envs    = n_envs,
            obs_shape = env_info["obs_shape"],
            n_actions = env_info["n_actions"],
        )

    def store(self, **step):

        if self.full:
            raise RuntimeError("RolloutBuffer is full.")

        for name, value in step.items():
            getattr(self, name)[self.ptr] = value
        self.ptr += 1


    def clear(self):
        self.ptr = 0
    # ------------------------------Z-------------------------
