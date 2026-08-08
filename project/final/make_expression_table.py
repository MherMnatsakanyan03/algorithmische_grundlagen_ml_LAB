r"""Generate a LaTeX table of the learned formulas (expression_table.tex).

Companion to the NMSE table: instead of error numbers, it shows what each
regressor actually learned, cleaned up for the report:

  * expression strings from results.csv are parsed into sympy (gplearn's
    prefix notation included), so numeric subexpressions such as
    div(0.659, 0.777) fold into plain numbers,
  * all constants are rounded to three significant digits,
  * additive constants below 1e-3 are dropped (least-squares residue),
  * hopeless cases (gplearn's bloated approximations) are abbreviated to
    a node count instead of an unreadable formula,
  * entries whose extrapolation NMSE exceeds 1e-3 are marked with a
    dagger.

Only the targets where the three methods actually differ are printed;
the ones every method recovers would be three identical rows each and
are listed in the caption instead. Set OMITTED = [] to print all nine.

Run:  python3 make_expression_table.py   ->  expression_table.tex
then \input{expression_table} in the report.
"""

import csv
import math
import re

import sympy as sp

RESULTS = "results.csv"
OUT = "expression_table.tex"
EXPERIMENT = "function_suite"

# row order + how the ground-truth target is typeset in the first column
TARGETS = [
    ("quadratic",      r"$0.5x^2 - x$"),
    ("high_freq_sine", r"$\sin 3x$"),
    ("hero",           r"$0.5\,x\sin x + 0.1x^2$"),
    ("exp_growth",     r"$e^{0.5x}$"),
    ("rational",       r"$1/(1+x^2)$"),
]

# recovered by all three methods -> named in the caption, not given a row
OMITTED = [
    ("linear",    r"$2x + 1$"),
    ("sine",      r"$\sin x$"),
    ("product",   r"$x \sin x$"),
    ("sqrt_like", r"$\sqrt{|x|}$"),
]

METHODS = [("XiGuidedSR", "A"), ("GplearnSR", "B"), ("NeuralSR", "C")]

XSYM = sp.Symbol("x")


# --------------------- expression string -> sympy --------------------------

def _protected(fn):
    """Apply |.| only to numeric negative arguments (so log(-0.104) folds to
    a real number, matching gplearn's protected semantics) while leaving
    symbolic arguments unwrapped for readable output. The caption states
    that the protection is suppressed in print."""
    def wrapped(a):
        a = sp.sympify(a)
        if a.is_number and a.is_negative:
            a = sp.Abs(a)
        return fn(a)
    return wrapped


# environment for evaluating gplearn's prefix notation, e.g. mul(sin(X0), X0)
GPLEARN_ENV = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    "sqrt": _protected(sp.sqrt),
    "log": _protected(sp.log),
    "neg": lambda a: -a,
    "X0": XSYM,
}


def parse_expression(method: str, expr: str):
    """Turn an expression string from results.csv into a sympy expression."""
    if method == "GplearnSR":
        return sp.sympify(eval(expr, {"__builtins__": {}}, GPLEARN_ENV))
    # XiGuidedSR / NeuralSR already store sympy-printable infix strings
    return sp.sympify(expr, locals={"x": XSYM})


_TOKEN = re.compile(r"[A-Za-z_]\w*|-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def count_program_nodes(expr_str: str) -> int:
    """Node count of gplearn's raw prefix program: every operator name,
    every X0 and every numeric constant counts as one node. This is the
    length gplearn itself reports and the number a reader gets by counting
    the printed program. Traversing the simplified sympy tree instead gives
    a smaller, harder-to-interpret figure, because parsing folds numeric
    subexpressions and cancels pairs such as exp(log(x))."""
    return len(_TOKEN.findall(expr_str))


# ------------------------------ cleanup ------------------------------------

def _round3(f):
    """Round a Float to 3 significant digits; values that round to an
    integer become Integers, so coefficients like 1.002*x print as x."""
    v = float(f)
    if v == 0:
        return sp.Integer(0)
    v = round(v, 2 - math.floor(math.log10(abs(v))))  # 3 significant digits
    if v == int(v):
        return sp.Integer(int(v))
    return sp.Float(v, 3)


def _drop_tiny(add_expr):
    """Drop pure-number summands below 1e-3 (fitting residue), unless that
    would delete the whole expression (e.g. a genuinely constant fit)."""
    kept = [t for t in add_expr.args
            if not (t.is_Number and abs(t) < 1e-3)]
    return sp.Add(*kept) if kept else add_expr


def clean(expr):
    expr = expr.xreplace({f: _round3(f) for f in expr.atoms(sp.Float)})
    expr = expr.replace(lambda e: e.is_Add, _drop_tiny)
    expr = expr.replace(   # |x|**0.5 -> sqrt(|x|)  (Float exponent)
        lambda e: e.is_Pow and e.exp.is_Float and float(e.exp) == 0.5,
        lambda e: sp.sqrt(e.base))
    return expr


def cell(expr, nmse: float, raw: str, method: str) -> str:
    """LaTeX for one table entry; abbreviate unprintable monsters and mark
    failed structures with a dagger."""
    tex = sp.latex(expr, fold_short_frac=True)
    if len(tex) > 240:
        if method == "GplearnSR":
            n_nodes = count_program_nodes(raw)
        else:
            n_nodes = sum(1 for _ in sp.preorder_traversal(expr))
        body = rf"\emph{{bloated approximation, {n_nodes} nodes}}"
    else:
        body = f"${tex}$"
    if nmse > 1e-3:   # did not extrapolate -> structure not recovered
        body += r"\,${}^{\dagger}$"
    return body


# ------------------------------ assembly -----------------------------------

def load_rows() -> dict:
    """(function, method) -> (expression string, nmse)."""
    out = {}
    with open(RESULTS, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["experiment"] == EXPERIMENT:
                out[(row["function"], row["method"])] = (
                    row["expression"], float(row["nmse"]))
    return out


def omitted_sentence() -> str:
    """Caption clause naming the targets that were left out."""
    if not OMITTED:
        return ""
    names = [tex for _, tex in OMITTED]
    listed = ", ".join(names[:-1]) + " and " + names[-1]
    return (f" The other {len(names)} targets ({listed}) are recovered by"
            r" all three methods and are omitted.")


def main():
    rows = load_rows()
    lines = [
        "% auto-generated by make_expression_table.py -- do not edit",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lll@{}}",
        r"\toprule",
        r"Target & & Learned formula \\",
        r"\midrule",
    ]
    for i, (fname, target_tex) in enumerate(TARGETS):
        for j, (method, label) in enumerate(METHODS):
            expr_str, nmse = rows[(fname, method)]
            expr = clean(parse_expression(method, expr_str))
            first_col = target_tex if j == 0 else ""
            lines.append(f"{first_col} & {label} & "
                         f"{cell(expr, nmse, expr_str, method)} \\\\")
        if i < len(TARGETS) - 1:
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Learned formulas on the targets where the methods"
        r" differ (same runs as Table~\ref{tab:suite})."
        + omitted_sentence() +
        r" Constants are rounded to three significant digits, numeric"
        r" subexpressions are folded, and additive constants below"
        r" $10^{-3}$ (least-squares residue) are dropped; gplearn's"
        r" protected operators are printed as plain $\log$,"
        r" $\sqrt{\cdot}$ and $\div$ for readability. Entries marked"
        r" $\dagger$ have extrapolation NMSE above $10^{-3}$ in"
        r" Table~\ref{tab:suite}.}",
        r"\label{tab:formulas}",
        r"\end{table}",
    ]
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(TARGETS)} targets, {len(OMITTED)} omitted)")


if __name__ == "__main__":
    main()