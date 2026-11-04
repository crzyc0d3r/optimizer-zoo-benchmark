import unittest

import numpy as np

from optimizer_zoo.benchmark import effective_time, run_benchmark, run_one
from optimizer_zoo.model import MLP, make_dataset
from optimizer_zoo.optimizers import (ALL_OPTIMIZERS, Adafactor, Adam, AdamW, Aux, Lion,
                                      Muon, Shampoo, SOAP, Sophia, newton_schulz)


def quadratic_grads(params):
    """Gradient of 0.5*||p||^2 for every parameter -> optimum at zero."""
    return {k: v.copy() for k, v in params.items()}


def fresh_params(seed=0):
    rng = np.random.default_rng(seed)
    return {"W1": rng.standard_normal((12, 8)).astype(np.float32),
            "b1": rng.standard_normal(8).astype(np.float32),
            "W2": rng.standard_normal((8, 4)).astype(np.float32),
            "b2": rng.standard_normal(4).astype(np.float32)}


class TestEveryOptimizerDescends(unittest.TestCase):
    def test_all_twelve_reduce_a_quadratic(self):
        self.assertEqual(len(ALL_OPTIMIZERS), 12)
        for cls in ALL_OPTIMIZERS:
            params = fresh_params()
            start = sum(float((v ** 2).sum()) for v in params.values())
            opt = cls(params)
            aux = Aux(sampled_grads=lambda p=params: quadratic_grads(p), batch_size=8)
            for _ in range(60):
                opt.step(params, quadratic_grads(params), aux)
            end = sum(float((v ** 2).sum()) for v in params.values())
            self.assertLess(end, start, f"{cls.name} did not descend")
            self.assertTrue(all(np.isfinite(v).all() for v in params.values()), cls.name)
            self.assertGreater(opt.state_bytes(), 0, cls.name)


class TestStateMemory(unittest.TestCase):
    def test_adam_is_two_buffers_and_momentum_one(self):
        params = fresh_params()
        n_bytes = sum(v.nbytes for v in params.values())
        for cls, factor in ((Adam, 2), (AdamW, 2), (Lion, 1)):
            opt = cls(params)
            opt.step(params, quadratic_grads(params))
            self.assertEqual(opt.state_bytes(), factor * n_bytes, cls.name)

    def test_adafactor_stores_rows_plus_columns_for_matrices(self):
        params = {"W": np.ones((40, 30), np.float32)}
        opt = Adafactor(params)
        opt.step(params, {"W": np.ones((40, 30), np.float32)})
        self.assertEqual(opt.state_bytes(), (40 + 30) * 4)  # not 40*30*4

    def test_shampoo_and_soap_store_square_statistics(self):
        params = {"W": np.ones((40, 30), np.float32)}
        for cls in (Shampoo, SOAP):
            opt = cls(params)
            opt.step(params, {"W": np.ones((40, 30), np.float32)})
            names = " ".join(opt.state_arrays())
            self.assertIn("(40, 40)", names, cls.name)
            self.assertIn("(30, 30)", names, cls.name)


class TestUpdateRules(unittest.TestCase):
    def test_lion_moves_every_coordinate_by_lr(self):
        rng = np.random.default_rng(1)
        params = {"W": rng.standard_normal((6, 6)).astype(np.float32)}
        before = params["W"].copy()
        g = rng.standard_normal((6, 6)).astype(np.float32) * np.array([1e-3, 1, 1e3, 1, 1, 1], np.float32)
        Lion(params, lr=0.01).step(params, {"W": g})
        self.assertTrue(np.allclose(np.abs(before - params["W"]), 0.01, atol=1e-6))

    def test_adamw_decay_is_independent_of_gradient_scale_but_l2_is_not(self):
        rng = np.random.default_rng(2)
        w = rng.standard_normal((5, 5)).astype(np.float32)
        shrink = {}
        for scale in (1e-3, 1.0):
            pa, pw = {"W": w.copy()}, {"W": w.copy()}
            a, b = Adam(pa, lr=1e-2, l2=0.1), AdamW(pw, lr=1e-2, weight_decay=0.1)
            g = np.full_like(w, scale)
            for _ in range(30):
                a.step(pa, {"W": g})
                b.step(pw, {"W": g})
            shrink[scale] = (np.abs(pa["W"]).mean(), np.abs(pw["W"]).mean())
        # AdamW's shrinkage does not depend on the gradient scale
        self.assertAlmostEqual(shrink[1e-3][1], shrink[1.0][1], places=5)
        # coupled L2 inside Adam changes with gradient scale
        self.assertNotAlmostEqual(shrink[1e-3][0], shrink[1.0][0], places=2)

    def test_newton_schulz_approximately_orthogonalises(self):
        rng = np.random.default_rng(3)
        G = rng.standard_normal((16, 10)).astype(np.float32)
        O = newton_schulz(G, steps=5)
        s = np.linalg.svd(O, compute_uv=False)
        self.assertTrue(np.all(s > 0.5) and np.all(s < 1.5), s)

    def test_muon_uses_adamw_for_vectors_and_last_layer(self):
        params = fresh_params()
        opt = Muon(params)
        self.assertEqual(opt.muon_params, {"W1"})
        opt.step(params, quadratic_grads(params))
        names = " ".join(opt.state_arrays())
        self.assertIn("W1.momentum", names)
        self.assertIn("W2.m", names)
        self.assertIn("b1.m", names)

    def test_sophia_clips_update_to_lr(self):
        params = {"W": np.ones((4, 4), np.float32) * 5}
        opt = Sophia(params, lr=0.1, hess_every=1)
        g = np.ones((4, 4), np.float32)
        aux = Aux(sampled_grads=lambda: {"W": np.full((4, 4), 1e-6, np.float32)}, batch_size=1)
        before = params["W"].copy()
        opt.step(params, {"W": g}, aux)
        self.assertTrue(np.allclose(before - params["W"], 0.1))


class TestBenchmark(unittest.TestCase):
    def test_tradeoff_arithmetic(self):
        self.assertAlmostEqual(effective_time(100, 1.0), 100.0)
        self.assertAlmostEqual(effective_time(80, 1.3), 104.0)

    def test_run_one_reports_steps_to_target(self):
        x, y = make_dataset(n=1024, seed=0)
        r = run_one(AdamW, 1e-2, x, y, max_steps=400, target_loss=0.6)
        self.assertIsNotNone(r.steps_to_target)
        self.assertGreater(r.ms_per_step, 0)
        self.assertEqual(r.state_bytes, 2 * sum(v.nbytes for v in MLP(24, 64, 6).params.values()))

    def test_benchmark_is_deterministic(self):
        a = run_benchmark(n=512, max_steps=40, optimizers=["Adam"], grids={"Adam": (1e-2,)})
        b = run_benchmark(n=512, max_steps=40, optimizers=["Adam"], grids={"Adam": (1e-2,)})
        self.assertEqual(a["results"][0]["final_loss"], b["results"][0]["final_loss"])


if __name__ == "__main__":
    unittest.main()
