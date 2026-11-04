"""CLI: ``python -m optimizer_zoo [bench|memory|decay|tradeoff]``."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import memory_report
from .benchmark import effective_time, format_table, run_benchmark
from .optimizers import ALL_OPTIMIZERS


def cmd_bench(args: argparse.Namespace) -> int:
    names = args.only.split(",") if args.only else None
    report = run_benchmark(max_steps=args.max_steps, target_loss=args.target_loss,
                           seed=args.seed, optimizers=names, hidden=args.hidden,
                           batch_size=args.batch_size)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print(format_table(report))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    print(memory_report.format_memory_table(args.params, args.rows, args.cols))
    return 0


def cmd_decay(args: argparse.Namespace) -> int:
    """Show why coupled L2 inside Adam is not the same as decoupled decay."""
    from .optimizers import Adam, AdamW
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 4)).astype(np.float32)
    g_big = np.full_like(w, 1.0)    # large, consistent gradients
    g_small = np.full_like(w, 1e-3)  # tiny gradients
    for label, g in (("large gradients", g_big), ("tiny gradients", g_small)):
        pa = {"W": w.copy()}
        pw = {"W": w.copy()}
        adam_l2 = Adam(pa, lr=1e-2, l2=0.1)
        adamw = AdamW(pw, lr=1e-2, weight_decay=0.1)
        for _ in range(50):
            adam_l2.step(pa, {"W": g})
            adamw.step(pw, {"W": g})
        print(f"{label:>16}: |w| after 50 steps  Adam+L2 = {np.abs(pa['W']).mean():.4f}   "
              f"AdamW = {np.abs(pw['W']).mean():.4f}   (start {np.abs(w).mean():.4f})")
    print("With L2 inside Adam the penalty is divided by sqrt(v) like the gradient, so its\n"
          "strength depends on gradient scale. AdamW shrinks the weights by lr*wd each step\n"
          "regardless of the gradient, which makes the regularisation strength predictable.")
    return 0


def cmd_tradeoff(args: argparse.Namespace) -> int:
    base = effective_time(args.base_steps, args.base_step_time)
    alt_steps = args.base_steps * (1 - args.fewer_steps)
    alt_time = args.base_step_time * (1 + args.slower_step)
    alt = effective_time(alt_steps, alt_time)
    print(f"baseline : {args.base_steps:.0f} steps x {args.base_step_time:.2f}s = {base:.1f}s")
    print(f"candidate: {alt_steps:.0f} steps x {alt_time:.2f}s = {alt:.1f}s  "
          f"({(alt / base - 1) * 100:+.1f}% wall clock)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="optimizer_zoo", description=__doc__)
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("bench", help="steps-to-target / ms-per-step / state-memory table")
    b.add_argument("--max-steps", type=int, default=600)
    b.add_argument("--target-loss", type=float, default=0.35)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--hidden", type=int, default=64)
    b.add_argument("--batch-size", type=int, default=128)
    b.add_argument("--only", help="comma-separated optimizer names: " + ",".join(c.name for c in ALL_OPTIMIZERS))
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_bench)

    m = sub.add_parser("memory", help="optimizer-state memory at a given parameter count")
    m.add_argument("--params", type=float, default=1e9, help="total parameter count")
    m.add_argument("--rows", type=int, default=4096, help="typical matrix rows (for Adafactor/Shampoo)")
    m.add_argument("--cols", type=int, default=4096, help="typical matrix cols")
    m.set_defaults(fn=cmd_memory)

    d = sub.add_parser("decay", help="coupled L2 in Adam vs decoupled decay in AdamW")
    d.set_defaults(fn=cmd_decay)

    t = sub.add_parser("tradeoff", help="fewer-steps vs slower-steps arithmetic")
    t.add_argument("--base-steps", type=float, default=100)
    t.add_argument("--base-step-time", type=float, default=1.0)
    t.add_argument("--fewer-steps", type=float, default=0.2, help="fraction fewer steps (0.2 = 20%%)")
    t.add_argument("--slower-step", type=float, default=0.3, help="fraction slower per step (0.3 = 30%%)")
    t.set_defaults(fn=cmd_tradeoff)

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 1
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
