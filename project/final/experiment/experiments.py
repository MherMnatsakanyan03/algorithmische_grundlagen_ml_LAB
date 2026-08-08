"""Evaluate all three regressors on the SAME datasets.

Every experiment builds one Dataset per (function, setting, trial-seed) and
feeds the identical arrays to all methods, so numbers are directly
comparable. Scores are reported as NMSE = MSE_test / Var(y_test), which is
scale-free (0 = perfect, ~1 = as bad as predicting the test mean), so it
can be compared and averaged across functions -- raw MSE cannot.

The test cases target the hypothesized strengths / weaknesses:

  function_suite   -- breadth over function types. Expectations:
                      * additive multi-term forms (hero) hurt XiGuidedSR
                        (xi saturates, see regressors.py docstring)
                      * high-frequency sine hurts NeuralSR (its fixed
                        64-point input grid aliases fast oscillations)
                      * fine-tuned constants hurt GplearnSR (gplearn has
                        no constant optimizer, only random ephemerals)
  noise_sweep      -- rank-based xi screening should degrade gracefully;
                      gplearn's raw-MSE fitness chases the noise
  band_sweep       -- extrapolation distance exposes wrong-but-similar
                      structures (NeuralSR's log-vs-sqrt confusion grows
                      with distance; in-domain fit quality does not)
  trainsize_sweep  -- Chatterjee xi is unreliable at small n (known
                      limitation) -> XiGuidedSR should suffer most at n=20

Run:  python3 experiments.py            (results printed + results.csv)
Set QUICK = True for a fast smoke run with tiny budgets.
"""

import csv
import os
import time

import numpy as np

from project.final.experiment.datagen import Node, generate_dataset, mse
from project.final.experiment.regressors import XiGuidedSR, GplearnSR
from project.final.experiment.neural_sr import NeuralSR

QUICK = False
TRIALS = [0]            # add seeds, e.g. [0, 1, 2], for mean-over-trials
WEIGHTS = "neural_sr.pt"


# ------------------------- test functions (ground truth) -------------------

def C(v): return Node("const", value=v)
def X(): return Node("x")
def u(op, a): return Node(op, [a])
def b(op, a, c): return Node(op, [a, c])


FUNCTIONS = {
    "linear":         b("add", b("mul", C(2.0), X()), C(1.0)),
    "quadratic":      b("add", b("mul", C(0.5), u("square", X())),
                        b("mul", C(-1.0), X())),
    "sine":           u("sin", X()),
    "high_freq_sine": u("sin", b("mul", C(3.0), X())),
    "product":        b("mul", X(), u("sin", X())),
    "hero":           b("add", b("mul", C(0.5), b("mul", X(), u("sin", X()))),
                        b("mul", C(0.1), u("square", X()))),
    "sqrt_like":      u("sqrt", X()),
    "exp_growth":     u("exp", b("mul", C(0.5), X())),
    "rational":       b("div", C(1.0), b("add", C(1.0), u("square", X()))),
}


# ------------------------------ harness ------------------------------------

_neural = None  # pretrained once, reused across all fits


def get_neural():
    global _neural
    if _neural is None:
        if os.path.exists(WEIGHTS):
            _neural = NeuralSR(weights_path=WEIGHTS)
        else:
            print(f"[{WEIGHTS} not found -> pretraining once, then saving]")
            _neural = NeuralSR()
            n = 2000 if QUICK else 20000
            _neural.pretrain(n_functions=n, epochs=2 if QUICK else 5,
                             save_path=WEIGHTS)
    return _neural


def make_regressors(seed):
    """Fresh search instances per dataset; the neural net is reused."""
    if QUICK:
        return {"XiGuidedSR": XiGuidedSR(n_iter=2000, seed=seed),
                "GplearnSR": GplearnSR(generations=5, population_size=300,
                                       seed=seed),
                "NeuralSR": get_neural()}
    return {"XiGuidedSR": XiGuidedSR(n_iter=20000, seed=seed),
            "GplearnSR": GplearnSR(seed=seed),
            "NeuralSR": get_neural()}


def evaluate(reg, ds):
    t0 = time.time()
    reg.fit(ds.x_train, ds.y_train)
    err = mse(reg.predict(ds.x_test), ds.y_test)
    # normalize over the full y range: Var(y_test) alone explodes NMSE for
    # functions with a nearly-constant test tail (e.g. 1/(1+x^2))
    denom = max(np.var(np.concatenate([ds.y_train, ds.y_test])), 1e-12)
    return err, err / denom, time.time() - t0, reg.get_expression()


def run(experiment, fname, param, results, **ds_kwargs):
    """One (function, setting): same dataset for every method, per trial."""
    f = FUNCTIONS[fname]
    for trial in TRIALS:
        ds = generate_dataset(f, seed=trial, **ds_kwargs)
        for name, reg in make_regressors(seed=trial).items():
            err, nmse, secs, expr = evaluate(reg, ds)
            results.append(dict(experiment=experiment, function=fname,
                                param=param, method=name, trial=trial,
                                mse=err, nmse=nmse, seconds=round(secs, 1),
                                expression=expr))
            print(f"  {fname:15s} {param:12s} {name:11s} "
                  f"NMSE={nmse:8.3f}  MSE={err:10.4f}  {secs:5.1f}s  "
                  f"{expr[:60]}")


def main():
    results = []
    noise = 0.01

    print("== 1. function suite (noise=0.01, band=3) ==")
    for fname in FUNCTIONS:
        run("function_suite", fname, "-", results, noise_std=noise)

    print("== 2. noise sweep ==")
    for fname in ("quadratic", "product"):
        for ns in (0.0, 0.05, 0.2):
            run("noise_sweep", fname, f"noise={ns}", results, noise_std=ns)

    print("== 3. extrapolation-distance sweep ==")
    for fname in ("sqrt_like", "hero"):
        for band in (1.0, 3.0, 6.0):
            run("band_sweep", fname, f"band={band}", results,
                noise_std=noise, test_band=band)

    print("== 4. training-size sweep (xi weak at small noisy n) ==")
    for n in (15, 50, 200):
        run("trainsize_sweep", "product", f"n={n}", results,
            noise_std=0.3, n_train=n)

    with open("results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n{len(results)} rows written to results.csv")

    # per-method NMSE summary over the function suite (comparable, scale-free)
    print("\n== summary: median NMSE over function suite ==")
    for m in ("XiGuidedSR", "GplearnSR", "NeuralSR"):
        vals = [r["nmse"] for r in results
                if r["experiment"] == "function_suite" and r["method"] == m]
        print(f"  {m:11s} median NMSE = {np.median(vals):.3f}")


if __name__ == "__main__":
    main()
