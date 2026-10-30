# ------------------------------Z-------------------------
import multiprocessing as mp
import traceback

import numpy as np


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
        self.remotes, self.processes = [], []
        for _ in range(self.n_envs):
            remote, worker_remote = mp.Pipe()
            p = mp.Process(target=ParallelRunner.worker, args=(worker_remote, make_env), daemon=True)
            p.start()
            # Closed before the next fork, so no other worker inherits this end:
            # if this worker dies, recv() raises EOFError instead of waiting forever
            worker_remote.close()
            self.remotes.append(remote)
            self.processes.append(p)

        self.remotes[0].send(("env_info", None))
        self.env_info = ParallelRunner.receive(self.remotes[0])

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
        except Exception: 
            remote.send(("error", traceback.format_exc()))  # let the main process raise instead of hanging
            raise
        finally:
            env.close()  # always close SC2, even after an error
            remote.close()

    @staticmethod
    def receive(remote):
        """
        Receive one answer, raising if the worker crashed.
        """
        answer = remote.recv()
        if isinstance(answer, tuple) and answer[0] == "error":
            raise RuntimeError(f"Environment worker crashed:\n{answer[1]}")
        return answer

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
        return ParallelRunner.stack([ParallelRunner.receive(r) for r in self.remotes])

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
        steps, finals, obs = zip(*[ParallelRunner.receive(r) for r in self.remotes])
        return ParallelRunner.stack(steps), ParallelRunner.stack(finals), ParallelRunner.stack(obs)

    def close(self):
        for r, p in zip(self.remotes, self.processes):
            if p.is_alive():
                try:
                    r.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
        for p in self.processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
# ------------------------------Z-------------------------