"""Analytic optimizer-state memory at scale.

The benchmark measures state on a toy MLP; this module extrapolates the same
accounting to a large model so the numbers match what you would see on a GPU
(e.g. Adam's two FP32 moment buffers at 1B parameters = 8 GB).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

FP32 = 4


def state_formulas(n_params: float, rows: int, cols: int) -> List[Tuple[str, float, str]]:
    """Return (optimizer, bytes, explanation) assuming FP32 state and that the
    model is made of ``rows x cols`` matrices."""
    n_mats = n_params / (rows * cols)
    out = [
        ("SGD+momentum", 1 * n_params * FP32, "1 buffer (velocity)"),
        ("AdaGrad", 1 * n_params * FP32, "1 buffer (sum of squared grads)"),
        ("RMSprop", 1 * n_params * FP32, "1 buffer (EMA of squared grads)"),
        ("Adam", 2 * n_params * FP32, "2 buffers (m, v)"),
        ("AdamW", 2 * n_params * FP32, "2 buffers (m, v); decay is stateless"),
        ("Adafactor", n_mats * (rows + cols) * FP32, "row + column stats per matrix (n+m, not n*m)"),
        ("Lion", 1 * n_params * FP32, "1 buffer (momentum); sign update needs no v"),
        ("AdEMAMix", 3 * n_params * FP32, "3 buffers (fast m, slow m, v)"),
        ("Shampoo", (1 * n_params + n_mats * (rows * rows + cols * cols) * 2) * FP32,
         "momentum + L (n*n), R (m*m) and their inverse roots per matrix"),
        ("SOAP", (2 * n_params + n_mats * (rows * rows + cols * cols) * 2) * FP32,
         "Adam m, v (rotated) + L, R and their eigenbases per matrix"),
        ("Muon", 1 * n_params * FP32, "1 buffer (momentum); Newton-Schulz is stateless"),
        ("Sophia", 2 * n_params * FP32, "2 buffers (m, diagonal Hessian h)"),
    ]
    return out


def format_memory_table(n_params: float, rows: int, cols: int) -> str:
    weights = n_params * FP32
    lines = [f"model: {n_params:,.0f} params -> weights {weights / 1e9:.2f} GB (FP32), "
             f"gradients another {weights / 1e9:.2f} GB; matrices assumed {rows}x{cols}",
             f"{'optimizer':<14}{'state':>10}{'x weights':>12}  what is stored",
             "-" * 70]
    for name, b, why in state_formulas(n_params, rows, cols):
        size = f"{b / 1e9:.2f} GB" if b >= 1e8 else f"{b / 1e6:.1f} MB"
        lines.append(f"{name:<14}{size:>10}{b / weights:>10.3f}x  {why}")
    return "\n".join(lines)
