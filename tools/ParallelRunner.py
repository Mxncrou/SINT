import multiprocessing as mp

import numpy as np


# ------------------------------Z-------------------------
class ParallelRunner:
    """
    E environments, each running in its own worker process.
    The main process sends orders through one pipe per environment and stacks the answers along a first axis of size E.
    """

    def __init__(self, make_env, n_envs):
        """
        Args:
            make_env (callable) : Function without argument that creates one environment.
            n_envs (int)        : Number of environments E.
        """
        self.n_envs = n_envs

        # One pipe per environment: the main process keeps one end, the worker gets the other
        self.remotes, worker_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.processes = [
            mp.Process(target=ParallelRunner.worker, args=(w, make_env), daemon=True)
            for w in worker_remotes
        ]

        for p in self.processes:
            p.start()
        for w in worker_remotes:
            w.close()  # the main process only keeps its own ends

        self.remotes[0].send(("env_info", None))
        self.env_info = self.remotes[0].recv()

    @staticmethod
    def observe(env):
        """
        Read what the agents need from one environment.
        Returns:
            dict : observations (N, obs_dim), available_actions (N, n_actions), alives (N,)
        """
        return {
            "observations": np.array(env.get_obs(), dtype=np.float32),
            "available_actions": np.array(env.get_avail_actions(), dtype=bool),
            "alives": np.array([env.get_unit_by_id(i).health > 0 for i in range(env.n_agents)]),
        }

    @staticmethod
    def worker(remote, make_env):
        """
        Run one environment in its own process: wait for an order, execute it, answer.
        Args:
            remote (Connection) : Worker end of the pipe.
            make_env (callable) : Function without argument that creates one environment.
        """
        env = make_env()
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == "env_info":
                    remote.send(env.get_env_info())
                elif cmd == "reset":
                    env.reset()
                    remote.send(ParallelRunner.observe(env))
                elif cmd == "step":
                    reward, terminated, info = env.step(data)
                    step = {
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": info.get("episode_limit", False),
                        "won": info.get("battle_won", False),
                    }
                    final = ParallelRunner.observe(env)  # o_{t+1}, before any reset
                    if terminated:
                        env.reset()
                        ob = ParallelRunner.observe(env)  # first observation of the next episode
                    else:
                        ob = final
                    remote.send((step, final, ob))
                elif cmd == "close":
                    break
        finally:
            env.close()  # always close SC2, even after an error
            remote.close()

    @staticmethod
    def stack(dicts):
        """
        E dictionaries of arrays (...) -> one dictionary of arrays (E, ...).
        """
        return {k: np.stack([d[k] for d in dicts]) for k in dicts[0]}

    def reset(self):
        """
        Returns:
            dict : Observations of the E environments, each array of shape (E, ...).
        """
        for r in self.remotes:
            r.send(("reset", None))
        return ParallelRunner.stack([r.recv() for r in self.remotes])

    def step(self, actions):
        """
        Send all orders first, then receive all answers, so the E environments play at the same time.
        Args:
            actions (np.ndarray) : Actions of all agents (E, N).
        Returns:
            step (dict)  : reward, terminated, truncated, won, each of shape (E,).
            final (dict) : Observations right after the step, before any reset, each array of shape (E, ...).
            ob (dict)    : Observations for the next step (first observation of a new episode after a reset).
        """
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        steps, finals, obs = zip(*[r.recv() for r in self.remotes])
        return ParallelRunner.stack(steps), ParallelRunner.stack(finals), ParallelRunner.stack(obs)

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.processes:
            p.join()
# ------------------------------Z-------------------------