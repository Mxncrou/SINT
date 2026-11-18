# ------------------------------Z-------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent / "runs" / "protoss"
TAG = "eval/winrate"
SMOOTH = 0.9


def ema(values, weight=SMOOTH):
    out, last = np.empty_like(values, dtype=float), 0.0
    for i, x in enumerate(values):
        last = last * weight + (1 - weight) * x
        out[i] = last / (1 - weight ** (i + 1))   # debiased, same as TensorBoard
    return out


def load_seed(seed_dir):
    ea = EventAccumulator(str(seed_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if TAG not in tags:
        raise KeyError(f"{TAG!r} not found in {seed_dir}\nAvailable: {tags}")
    by_step = {e.step: e.value for e in ea.Scalars(TAG)}  # later file wins on duplicate steps
    steps = np.array(sorted(by_step))
    values = np.array([by_step[s] for s in steps])
    return steps, ema(values)


fig, ax = plt.subplots(figsize=(9, 5))

for run_dir in sorted(ROOT.glob("*/*/*")):          # <model>/<algo>/<hash>
    seeds = sorted(run_dir.glob("seed_*"))
    if not seeds:
        continue
    curves = [load_seed(s) for s in seeds]

    # common step grid, limited to the range all seeds reached
    start = max(st[0] for st, _ in curves)
    end = min(st[-1] for st, _ in curves)
    grid = np.linspace(start, end, 500)
    Y = np.stack([np.interp(grid, st, v) for st, v in curves])

    mean = Y.mean(axis=0)
    std = Y.std(axis=0, ddof=1)

    model, algo = run_dir.parts[-3], run_dir.parts[-2]
    line, = ax.plot(grid, mean, label=f"{model} {algo}")
    ax.fill_between(grid, mean - std, mean + std, color=line.get_color(), alpha=0.2)

ax.set_xlabel("Environment steps")
ax.set_ylabel("Greedy winrate evaluation")
ax.set_ylim(0, 1)            # remove if winrate is in %, not [0, 1]
ax.grid(alpha=0.3)
ax.legend()
ax.set_title(f"Mean ± std (EMA smoothing = {SMOOTH})")
fig.tight_layout()
fig.savefig("rewards_comparison.png", dpi=200)
plt.show()
# ------------------------------Z-------------------------
