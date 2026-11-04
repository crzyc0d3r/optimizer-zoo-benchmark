"""Twelve neural-network optimizers implemented from scratch in NumPy.

Every optimizer shares one interface::

    opt = Adam(params, lr=1e-3)
    opt.step(params, grads, aux=None)   # updates ``params`` in place
    opt.state_bytes()                   # bytes of optimizer state held

so the benchmark can treat them uniformly and measure what each one *stores*
and how it *transforms the gradient*. The implementations follow the
published update rules; they are written for clarity, not for speed.

Family overview
---------------
* Smoothing / adaptive step size:  SGD+momentum, AdaGrad, RMSprop, Adam
* Regularization / memory / history:  AdamW, Adafactor, Lion, AdEMAMix
* Matrix structure / curvature:  Shampoo, SOAP, Muon, Sophia
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

Params = Dict[str, np.ndarray]


class Aux:
    """Extra information some optimizers need beyond the gradient.

    ``sampled_grads`` returns gradients against labels sampled from the model
    (used by Sophia); ``batch_size`` is the minibatch size of that gradient.
    """

    def __init__(self, sampled_grads: Optional[Callable[[], Params]] = None,
                 batch_size: int = 1) -> None:
        self.sampled_grads = sampled_grads
        self.batch_size = batch_size


class Optimizer:
    name = "base"

    def __init__(self, params: Params, lr: float, weight_decay: float = 0.0) -> None:
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.state: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in params}

    # -- helpers -------------------------------------------------------------
    def _zeros_like(self, name: str, key: str, ref: np.ndarray) -> np.ndarray:
        st = self.state[name]
        if key not in st:
            st[key] = np.zeros_like(ref, dtype=np.float32)
        return st[key]

    def state_bytes(self) -> int:
        total = 0
        for st in self.state.values():
            for v in st.values():
                if isinstance(v, np.ndarray):
                    total += v.nbytes
        return int(total)

    def state_arrays(self) -> List[str]:
        out = []
        for name, st in self.state.items():
            for key, v in st.items():
                if isinstance(v, np.ndarray):
                    out.append(f"{name}.{key}{tuple(v.shape)}")
        return out

    def step(self, params: Params, grads: Params, aux: Optional[Aux] = None) -> None:
        self.t += 1
        for name, p in params.items():
            self.update(name, p, grads[name].astype(np.float32), aux)

    def update(self, name: str, p: np.ndarray, g: np.ndarray, aux: Optional[Aux]) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Smoothing gradients and adapting step sizes
# ---------------------------------------------------------------------------
class SGDMomentum(Optimizer):
    """Running accumulation of gradients: persistent directions build up,
    inconsistent ones cancel. One parameter-sized buffer."""
    name = "SGD+momentum"

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum

    def update(self, name, p, g, aux):
        v = self._zeros_like(name, "v", p)
        v *= self.momentum
        v += g
        p -= self.lr * (v + self.weight_decay * p)


class AdaGrad(Optimizer):
    """Per-parameter step sizes from the *sum* of squared gradients. The sum
    only grows, so effective learning rates shrink toward zero."""
    name = "AdaGrad"

    def __init__(self, params, lr=1e-2, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.eps = eps

    def update(self, name, p, g, aux):
        G = self._zeros_like(name, "G", p)
        G += g * g
        p -= self.lr * (g / (np.sqrt(G) + self.eps) + self.weight_decay * p)


class RMSprop(Optimizer):
    """AdaGrad's sum replaced by an exponential moving average, so old
    squared gradients fade and the scaling tracks recent gradients."""
    name = "RMSprop"

    def __init__(self, params, lr=1e-3, rho=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.rho, self.eps = rho, eps

    def update(self, name, p, g, aux):
        E = self._zeros_like(name, "E", p)
        E *= self.rho
        E += (1.0 - self.rho) * g * g
        p -= self.lr * (g / (np.sqrt(E) + self.eps) + self.weight_decay * p)


class Adam(Optimizer):
    """Momentum (first moment) plus RMSprop-style second-moment scaling, both
    bias-corrected. Two parameter-sized buffers.

    ``l2`` adds a *coupled* L2 penalty to the gradient, i.e. the penalty is
    divided by sqrt(v) like everything else. Compare with AdamW below.
    """
    name = "Adam"

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, l2=0.0):
        super().__init__(params, lr, 0.0)
        self.b1, self.b2 = betas
        self.eps = eps
        self.l2 = l2

    def update(self, name, p, g, aux):
        if self.l2:
            g = g + self.l2 * p
        m = self._zeros_like(name, "m", p)
        v = self._zeros_like(name, "v", p)
        m *= self.b1
        m += (1 - self.b1) * g
        v *= self.b2
        v += (1 - self.b2) * g * g
        m_hat = m / (1 - self.b1 ** self.t)
        v_hat = v / (1 - self.b2 ** self.t)
        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Changing regularization, memory, and gradient history
# ---------------------------------------------------------------------------
class AdamW(Adam):
    """Adam with *decoupled* weight decay: the shrinkage is applied to the
    weights directly instead of flowing through the adaptive scaling."""
    name = "AdamW"

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, l2=0.0)
        self.weight_decay = weight_decay

    def update(self, name, p, g, aux):
        if self.weight_decay and p.ndim >= 2:
            p *= (1.0 - self.lr * self.weight_decay)
        super().update(name, p, g, aux)


class Adafactor(Optimizer):
    """Factored second moment: for an n x m matrix keep a row vector (n) and a
    column vector (m) of squared-gradient statistics instead of n*m values.
    Vectors keep a full second moment. No first-moment buffer by default."""
    name = "Adafactor"

    def __init__(self, params, lr=1e-2, beta2=0.999, eps=1e-30, clip=1.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta2, self.eps, self.clip = beta2, eps, clip

    def update(self, name, p, g, aux):
        b2 = 1.0 - self.t ** (-0.8) if self.t > 1 else 0.0  # Adafactor's growing beta2
        b2 = min(b2, self.beta2)
        g2 = g * g + self.eps
        if p.ndim == 2:
            R = self._zeros_like(name, "R", p[:, 0])  # row stats (n,)
            C = self._zeros_like(name, "C", p[0, :])  # column stats (m,)
            R *= b2
            R += (1 - b2) * g2.mean(axis=1)
            C *= b2
            C += (1 - b2) * g2.mean(axis=0)
            v_hat = np.outer(R / R.mean(), C)
        else:
            V = self._zeros_like(name, "V", p)
            V *= b2
            V += (1 - b2) * g2
            v_hat = V
        u = g / np.sqrt(v_hat)
        rms = np.sqrt((u * u).mean())
        u /= max(1.0, rms / self.clip)  # update clipping
        p -= self.lr * (u + self.weight_decay * p)


class Lion(Optimizer):
    """Momentum only; the update is the *sign* of a momentum/gradient blend,
    so every coordinate moves by the same magnitude lr. Needs a smaller
    learning rate than AdamW (typically 3-10x smaller)."""
    name = "Lion"

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.b1, self.b2 = betas

    def update(self, name, p, g, aux):
        m = self._zeros_like(name, "m", p)
        u = np.sign(self.b1 * m + (1 - self.b1) * g)
        m *= self.b2
        m += (1 - self.b2) * g
        p -= self.lr * (u + self.weight_decay * p)


class AdEMAMix(Optimizer):
    """Adam plus a second, *slow* EMA of gradients (beta3 ~ 0.9999) mixed into
    the numerator with weight alpha. Recent gradients stay influential while
    old directions survive much longer. Three parameter-sized buffers."""
    name = "AdEMAMix"

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=5.0,
                 eps=1e-8, weight_decay=0.0, warmup=200):
        super().__init__(params, lr, weight_decay)
        self.b1, self.b2, self.b3_final = betas
        self.alpha_final, self.eps, self.warmup = alpha, eps, warmup

    def _scheduled(self):
        frac = min(1.0, self.t / max(1, self.warmup))
        alpha = self.alpha_final * frac
        # beta3 warms up in log space as in the paper's scheduler
        b3_start = self.b1
        ln = np.log(b3_start) * (1 - frac) + np.log(self.b3_final) * frac
        return alpha, float(np.exp(ln))

    def update(self, name, p, g, aux):
        alpha, b3 = self._scheduled()
        m1 = self._zeros_like(name, "m1", p)
        m2 = self._zeros_like(name, "m2", p)
        v = self._zeros_like(name, "v", p)
        m1 *= self.b1
        m1 += (1 - self.b1) * g
        m2 *= b3
        m2 += (1 - b3) * g
        v *= self.b2
        v += (1 - self.b2) * g * g
        m1_hat = m1 / (1 - self.b1 ** self.t)
        v_hat = v / (1 - self.b2 ** self.t)
        p -= self.lr * ((m1_hat + alpha * m2) / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p)


# ---------------------------------------------------------------------------
# Using matrix structure and curvature
# ---------------------------------------------------------------------------
def _inv_root(mat: np.ndarray, power: float, eps: float) -> np.ndarray:
    """``(mat + eps I)^(-power)`` for a symmetric PSD matrix via eigendecomposition."""
    w, q = np.linalg.eigh(mat.astype(np.float64))
    w = np.maximum(w, 0.0) + eps
    return (q * w ** (-power)) @ q.T


class Shampoo(Optimizer):
    """Full-matrix preconditioning from both sides of a matrix gradient:
    ``L = EMA(G G^T)``, ``R = EMA(G^T G)``, update ``L^-1/4 G R^-1/4``.
    Stores n x n + m x m statistics per n x m matrix plus momentum. Vectors
    fall back to Adam. The preconditioner inverse roots are refreshed every
    ``precond_every`` steps because they need an eigendecomposition."""
    name = "Shampoo"

    def __init__(self, params, lr=1e-2, beta=0.95, momentum=0.9, eps=1e-6,
                 precond_every=10, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta, self.momentum, self.eps, self.precond_every = beta, momentum, eps, precond_every
        self._adam = Adam(params, lr=lr)

    def update(self, name, p, g, aux):
        if p.ndim != 2:
            self._adam.t = self.t
            self._adam.update(name, p, g, aux)
            self.state[name] = self._adam.state[name]
            return
        n, m = p.shape
        L = self._zeros_like(name, "L", np.zeros((n, n), np.float32))
        R = self._zeros_like(name, "R", np.zeros((m, m), np.float32))
        L *= self.beta
        L += (1 - self.beta) * (g @ g.T)
        R *= self.beta
        R += (1 - self.beta) * (g.T @ g)
        st = self.state[name]
        if "Linv" not in st or self.t % self.precond_every == 1 or self.precond_every == 1:
            st["Linv"] = _inv_root(L, 0.25, self.eps).astype(np.float32)
            st["Rinv"] = _inv_root(R, 0.25, self.eps).astype(np.float32)
        pre = st["Linv"] @ g @ st["Rinv"]
        # grafting: give the preconditioned direction the gradient's norm
        pre *= (np.linalg.norm(g) / (np.linalg.norm(pre) + 1e-12))
        v = self._zeros_like(name, "v", p)
        v *= self.momentum
        v += pre
        p -= self.lr * (v + self.weight_decay * p)


class SOAP(Optimizer):
    """Shampoo's preconditioners define an eigenbasis; rotate the gradient
    into it, run Adam *there*, rotate the update back. The basis is refreshed
    every ``precond_every`` steps. Stores Shampoo stats + eigenbases + Adam
    moments. Vectors fall back to Adam."""
    name = "SOAP"

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=0.95, eps=1e-8,
                 precond_every=10, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.b1, self.b2 = betas
        self.sb, self.eps, self.precond_every = shampoo_beta, eps, precond_every
        self._adam = Adam(params, lr=lr, betas=betas, eps=eps)

    def update(self, name, p, g, aux):
        if p.ndim != 2:
            self._adam.t = self.t
            self._adam.update(name, p, g, aux)
            self.state[name] = self._adam.state[name]
            return
        n, m = p.shape
        L = self._zeros_like(name, "L", np.zeros((n, n), np.float32))
        R = self._zeros_like(name, "R", np.zeros((m, m), np.float32))
        L *= self.sb
        L += (1 - self.sb) * (g @ g.T)
        R *= self.sb
        R += (1 - self.sb) * (g.T @ g)
        st = self.state[name]
        if "QL" not in st or self.t % self.precond_every == 1 or self.precond_every == 1:
            st["QL"] = np.linalg.eigh(L.astype(np.float64))[1].astype(np.float32)
            st["QR"] = np.linalg.eigh(R.astype(np.float64))[1].astype(np.float32)
        QL, QR = st["QL"], st["QR"]
        g_rot = QL.T @ g @ QR
        m_ = self._zeros_like(name, "m", p)
        v_ = self._zeros_like(name, "v", p)
        m_ *= self.b1
        m_ += (1 - self.b1) * g_rot
        v_ *= self.b2
        v_ += (1 - self.b2) * g_rot * g_rot
        m_hat = m_ / (1 - self.b1 ** self.t)
        v_hat = v_ / (1 - self.b2 ** self.t)
        upd_rot = m_hat / (np.sqrt(v_hat) + self.eps)
        upd = QL @ upd_rot @ QR.T
        if self.weight_decay:
            p *= (1 - self.lr * self.weight_decay)
        p -= self.lr * upd


def newton_schulz(G: np.ndarray, steps: int = 5) -> np.ndarray:
    """Approximate orthogonalisation of a matrix (the polar factor) using the
    quintic Newton-Schulz iteration from the Muon reference implementation."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float32)
    X = X / (np.linalg.norm(X) + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(Optimizer):
    """Momentum for hidden weight *matrices* is orthogonalised with
    Newton-Schulz iterations, which evens out the singular values of the
    update instead of scaling each coordinate. Embeddings, output layers and
    all vector parameters use AdamW. The ``muon_params`` set names which
    matrices Muon owns (default: every 2-D parameter except the last layer)."""
    name = "Muon"

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_lr=1e-3, weight_decay=0.0, muon_params: Optional[Iterable[str]] = None):
        super().__init__(params, lr, weight_decay)
        self.momentum, self.nesterov, self.ns_steps = momentum, nesterov, ns_steps
        matrices = [k for k, v in params.items() if v.ndim == 2]
        self.muon_params = set(muon_params) if muon_params is not None else set(matrices[:-1])
        self._adamw = AdamW(params, lr=adamw_lr, weight_decay=weight_decay)

    def update(self, name, p, g, aux):
        if name not in self.muon_params or p.ndim != 2:
            self._adamw.t = self.t
            self._adamw.update(name, p, g, aux)
            self.state[name] = self._adamw.state[name]
            return
        buf = self._zeros_like(name, "momentum", p)
        buf *= self.momentum
        buf += g
        u = g + self.momentum * buf if self.nesterov else buf
        O = newton_schulz(u, self.ns_steps)
        O *= max(1.0, p.shape[0] / p.shape[1]) ** 0.5
        p -= self.lr * (O + self.weight_decay * p)


class Sophia(Optimizer):
    """Second-order clipped optimizer (Sophia-G). Momentum is divided by a
    diagonal Hessian estimate ``h`` and the result is clipped elementwise to
    ``[-1, 1]``. ``h`` comes from the Gauss-Newton-Bartlett estimator
    (``B * g_hat^2`` with labels sampled from the model) and is refreshed only
    every ``hess_every`` steps, which keeps the average overhead small."""
    name = "Sophia"

    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99), rho=0.04, eps=1e-12,
                 hess_every=10, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.b1, self.b2, self.rho, self.eps, self.hess_every = betas[0], betas[1], rho, eps, hess_every
        self._pending_h: Optional[Params] = None

    def step(self, params, grads, aux=None):
        self.t += 1
        self._pending_h = None
        if aux is not None and aux.sampled_grads is not None and (self.t % self.hess_every == 1 or self.hess_every == 1):
            gs = aux.sampled_grads()
            # h_hat = g_hat^2; the batch-size factor B is applied in ``update``
            self._pending_h = {k: v.astype(np.float32) ** 2 for k, v in gs.items()}
        for name, p in params.items():
            self.update(name, p, grads[name].astype(np.float32), aux)

    def update(self, name, p, g, aux):
        m = self._zeros_like(name, "m", p)
        h = self._zeros_like(name, "h", p)
        m *= self.b1
        m += (1 - self.b1) * g
        if self._pending_h is not None:
            h *= self.b2
            h += (1 - self.b2) * self._pending_h[name]
        ratio = m / np.maximum(self.rho * aux_batch(aux) * h, self.eps)
        ratio = np.clip(ratio, -1.0, 1.0)
        p -= self.lr * (ratio + self.weight_decay * p)


def aux_batch(aux: Optional[Aux]) -> int:
    return aux.batch_size if aux is not None else 1


ALL_OPTIMIZERS = [SGDMomentum, AdaGrad, RMSprop, Adam, AdamW, Adafactor, Lion,
                  AdEMAMix, Shampoo, SOAP, Muon, Sophia]
BY_NAME = {cls.name: cls for cls in ALL_OPTIMIZERS}
