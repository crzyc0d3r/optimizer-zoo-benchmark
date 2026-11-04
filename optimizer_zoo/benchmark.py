"""Benchmark harness: steps-to-target, wall-clock per step, and optimizer
state memory for every optimizer on the same data, model and protocol.

The protocol is the one you should use on your own workload:

1. Same data, same model init, same minibatch order for every optimizer.
2. Each optimizer gets a learning-rate search over a small log grid.
3. Report *both* steps to a target quality and the wall-clock cost of those
   steps, because an optimizer that needs 20% fewer steps but makes each
   step 30% slower is ~4% slower overall.
4. Report peak optimizer-state memory next to the model's own size.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .model import MLP, make_dataset, n_params, param_bytes
from .optimizers import ALL_OPTIMIZERS, Aux, Optimizer

# Learning-rate grids. Sign-based and orthogonalised updates have a different
# natural scale, so copying AdamW's learning rate into them is not a fair
# comparison; each family gets a grid centred on its usual range.
DEFAULT_GRIDS: Dict[str, Sequence[float]] = {
    "SGD+momentum": (3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
    "AdaGrad": (3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
    "RMSprop": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "Adam": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "AdamW": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "Adafactor": (3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
    "Lion": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
    "AdEMAMix": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "Shampoo": (3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
    "SOAP": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "Muon": (3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
    "Sophia": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
}


@dataclass
class RunResult:
    optimizer: str
    lr: float
    steps_to_target: Optional[int]
    final_loss: float
    final_accuracy: float
    seconds: float
    ms_per_step: float
    state_bytes: int
    state_arrays: List[str] = field(default_factory=list)
    loss_curve: List[float] = field(default_factory=list)

    @property
    def reached(self) -> bool:
        return self.steps_to_target is not None

    def time_to_target(self) -> Optional[float]:
        if self.steps_to_target is None:
            return None
        return self.steps_to_target * self.ms_per_step / 1000.0


def effective_time(steps: float, seconds_per_step: float) -> float:
    """Wall-clock cost of a run: fewer steps only help if each step is not
    proportionally slower. 80 steps at 1.3 s = 104 s loses to 100 x 1.0 s."""
    return steps * seconds_per_step


def run_one(opt_cls, lr: float, x: np.ndarray, y: np.ndarray, *, hidden: int = 64,
            batch_size: int = 128, max_steps: int = 600, target_loss: float = 0.35,
            seed: int = 0, record_curve: bool = False, **opt_kwargs) -> RunResult:
    """Train one optimizer from a fixed init on a fixed minibatch schedule."""
    rng = np.random.default_rng(seed)
    model = MLP(x.shape[1], hidden, int(y.max()) + 1, seed=seed)
    opt: Optimizer = opt_cls(model.params, lr=lr, **opt_kwargs)
    order = rng.permutation(len(x))
    n_batches = len(x) // batch_size
    curve: List[float] = []
    steps_to_target: Optional[int] = None
    peak_state = 0
    sampler_rng = np.random.default_rng(seed + 1)

    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        idx = order[((step - 1) % n_batches) * batch_size:((step - 1) % n_batches + 1) * batch_size]
        xb, yb = x[idx], y[idx]
        loss, grads = model.loss_and_grads(xb, yb)
        aux = Aux(sampled_grads=lambda xb=xb: model.sampled_label_grads(xb, sampler_rng),
                  batch_size=batch_size)
        opt.step(model.params, grads, aux)
        peak_state = max(peak_state, opt.state_bytes())
        if record_curve:
            curve.append(loss)
        if steps_to_target is None and step % 5 == 0:
            if model.loss(x, y) <= target_loss:
                steps_to_target = step
                break
        if not np.isfinite(loss):
            break
    elapsed = time.perf_counter() - t0
    n_steps = step
    return RunResult(
        optimizer=opt.name, lr=lr, steps_to_target=steps_to_target,
        final_loss=model.loss(x, y), final_accuracy=model.accuracy(x, y),
        seconds=elapsed, ms_per_step=1000.0 * elapsed / max(1, n_steps),
        state_bytes=peak_state, state_arrays=opt.state_arrays(), loss_curve=curve,
    )


def lr_search(opt_cls, x, y, grid: Sequence[float], **kw) -> RunResult:
    """Pick the learning rate with the fewest steps to target; if none reach
    it, the one with the lowest final loss."""
    results = [run_one(opt_cls, lr, x, y, **kw) for lr in grid]
    reached = [r for r in results if r.reached and np.isfinite(r.final_loss)]
    if reached:
        return min(reached, key=lambda r: (r.steps_to_target, r.final_loss))
    finite = [r for r in results if np.isfinite(r.final_loss)]
    return min(finite or results, key=lambda r: r.final_loss)


def run_benchmark(*, n: int = 4096, d_in: int = 24, n_classes: int = 6, hidden: int = 64,
                  batch_size: int = 128, max_steps: int = 600, target_loss: float = 0.35,
                  seed: int = 0, optimizers: Optional[Sequence[str]] = None,
                  grids: Optional[Dict[str, Sequence[float]]] = None) -> Dict:
    x, y = make_dataset(n=n, d_in=d_in, n_classes=n_classes, seed=seed)
    grids = {**DEFAULT_GRIDS, **(grids or {})}
    chosen = [c for c in ALL_OPTIMIZERS if optimizers is None or c.name in optimizers]
    model_bytes = param_bytes(MLP(d_in, hidden, n_classes).params)
    results: List[RunResult] = []
    for cls in chosen:
        r = lr_search(cls, x, y, grids[cls.name], hidden=hidden, batch_size=batch_size,
                      max_steps=max_steps, target_loss=target_loss, seed=seed)
        results.append(r)
    return {
        "config": dict(n=n, d_in=d_in, n_classes=n_classes, hidden=hidden, batch_size=batch_size,
                       max_steps=max_steps, target_loss=target_loss, seed=seed,
                       n_params=n_params(MLP(d_in, hidden, n_classes).params),
                       model_bytes=model_bytes),
        "results": [asdict(r) for r in results],
    }


def format_table(report: Dict) -> str:
    cfg = report["config"]
    rows = report["results"]
    head = (f"{'optimizer':<14}{'best lr':>9}{'steps->target':>15}{'ms/step':>9}"
            f"{'time->target':>14}{'state bytes':>13}{'x model':>9}{'final loss':>12}{'acc':>7}")
    lines = [head, "-" * len(head)]
    baseline = next((r for r in rows if r["optimizer"] == "AdamW"), rows[0])
    for r in rows:
        steps = r["steps_to_target"]
        steps_s = str(steps) if steps is not None else f">{cfg['max_steps']}"
        ttt = (steps * r["ms_per_step"] / 1000.0) if steps is not None else None
        ttt_s = f"{ttt:.3f}s" if ttt is not None else "n/a"
        ratio = r["state_bytes"] / max(1, cfg["model_bytes"])
        lines.append(f"{r['optimizer']:<14}{r['lr']:>9.0e}{steps_s:>15}{r['ms_per_step']:>9.3f}"
                     f"{ttt_s:>14}{r['state_bytes']:>13,d}{ratio:>8.2f}x{r['final_loss']:>12.4f}"
                     f"{r['final_accuracy']:>7.3f}")
    lines.append("")
    lines.append(f"model: {cfg['n_params']:,} params = {cfg['model_bytes']:,} bytes (fp32); "
                 f"target train loss <= {cfg['target_loss']}; batch {cfg['batch_size']}; "
                 f"max {cfg['max_steps']} steps; best lr chosen per optimizer by steps-to-target.")
    b_steps, b_ms = baseline["steps_to_target"], baseline["ms_per_step"]
    if b_steps:
        lines.append(f"reference ({baseline['optimizer']}): {b_steps} steps x {b_ms:.3f} ms = "
                     f"{effective_time(b_steps, b_ms/1000):.3f}s. Fewer steps only win if the "
                     f"per-step cost does not grow faster than the step count shrinks.")
    return "\n".join(lines)
