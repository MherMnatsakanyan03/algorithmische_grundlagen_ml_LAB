"""Plot a successful recovery as the companion piece to hero_plot.pdf.

hero_plot.pdf shows the failure case (all three structures wrong, one of
them unbounded). This script shows the flip side on x*sin(x) from the
function suite: all three regressors recover the exact structure, so
after constant fitting their predictions sit on the true curve even far
outside the training interval.

The fitted expressions are read from results.csv (function_suite /
product), so the figure always reflects the actual experimental output
rather than hand-typed formulas.

Run:  python3 plot_good_fit.py   ->  good_fit_plot.pdf
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from project.final.experiment.datagen import Node, generate_dataset

RESULTS = "results.csv"
OUT = "good_fit_plot.pdf"

# which row of results.csv to visualize: the product function x*sin(x)
EXPERIMENT, FUNCTION = "function_suite", "product"

# ground truth, identical to FUNCTIONS["product"] in experiments.py
TRUE_TREE = Node("mul", [Node("x"), Node("sin", [Node("x")])])

XSYM = sp.Symbol("x")  # the single symbol every parsed expression uses

# gplearn stores prefix notation like mul(sin(X0), X0); evaluating the
# string against these (protected, matching gplearn's semantics) sympy
# building blocks turns it directly into a sympy expression
GPLEARN_ENV = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    "sqrt": lambda a: sp.sqrt(sp.Abs(a)),  # gplearn's protected sqrt
    "log": lambda a: sp.log(sp.Abs(a)),    # gplearn's protected log
    "neg": lambda a: -a,
    "X0": XSYM,
}


def parse_expression(method: str, expr: str):
    """Turn an expression string from results.csv into a sympy expression."""
    if method == "GplearnSR":
        # prefix notation -> sympy via the environment above
        return sp.sympify(eval(expr, {"__builtins__": {}}, GPLEARN_ENV))
    # XiGuidedSR / NeuralSR already store sympy-printable infix strings
    return sp.sympify(expr, locals={"x": XSYM})


def load_expressions() -> dict:
    """method -> fitted expression string, from the chosen results.csv rows."""
    out = {}
    with open(RESULTS, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["experiment"] == EXPERIMENT and row["function"] == FUNCTION:
                out[row["method"]] = row["expression"]
    if not out:
        raise SystemExit(f"no rows for {EXPERIMENT}/{FUNCTION} in {RESULTS}")
    return out


def main():
    # same dataset settings as the function suite (noise 0.01, seed 0, band 3)
    ds = generate_dataset(TRUE_TREE, noise_std=0.01, seed=0)

    xs = np.linspace(-8.0, 8.0, 1000)     # training interval plus both test bands
    y_true = TRUE_TREE.evaluate(xs)

    # per-method plot style: (linestyle, color, legend label)
    styles = {
        "XiGuidedSR": ("--", "tab:blue",   r"A: $\xi$-guided"),
        "GplearnSR":  (":",  "tab:orange", "B: gplearn"),
        "NeuralSR":   ("-.", "tab:green",  "C: neural"),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axvspan(-5, 5, color="0.92", zorder=0)  # shaded training interval
    ax.scatter(ds.x_train, ds.y_train, s=8, color="0.55", alpha=0.6,
               zorder=2, label="training points (noisy)")
    ax.plot(xs, y_true, "k-", lw=2.5, zorder=3,
            label=r"true $f(x) = x\,\sin x$")

    for method, expr_str in load_expressions().items():
        f = sp.lambdify(XSYM, parse_expression(method, expr_str), "numpy")
        y = np.asarray(f(xs), dtype=float)
        if y.ndim == 0:                    # constant expression -> broadcast
            y = np.full_like(xs, float(y))
        ls, color, label = styles[method]
        ax.plot(xs, y, ls, color=color, lw=1.8, zorder=4, label=label)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_xlim(-8, 8)
    # legend below the axes so it never covers the curves
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()