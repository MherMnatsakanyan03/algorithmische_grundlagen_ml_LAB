"""The common regressor interface and two of the three implementations:

  A) XiGuidedSR   -- random-mutation search over expression trees where the
                     Chatterjee xi coefficient (scipy.stats.chatterjeexi)
                     decides whether a "roll" is kept, i.e. how the search
                     continues. Constants are only fitted at the very end.
  B) GplearnSR    -- conventional genetic-programming symbolic regression,
                     thin wrapper around the gplearn library (the classic
                     GP approach surveyed in Dong & Zhong 2025; same family
                     as PySR, Cranmer 2023).

The third implementation (neural network that predicts a formula) lives in
neural_sr.py because it needs its own pretraining step.

Why xi: xi(g(X), y) -> 1 iff y is a (any, incl. non-monotonic) function of
g(X) (Chatterjee 2021). So a candidate g that has the right *structure* gets
a high score even before its constants / an affine rescaling are fitted,
and xi's rank-based nature makes the score robust to noise.
"""

import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chatterjeexi

from datagen import Node, FunctionSampler, UNARY, BINARY


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Regressor:
    """fit on (features, labels), predict labels for new features."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Regressor":
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_expression(self) -> str:
        """Human-readable formula of the fitted model (if it has one)."""
        return "n/a"


# ---------------------------------------------------------------------------
# Shared helpers for tree-based regressors
# ---------------------------------------------------------------------------

def _all_nodes(tree: Node) -> list:
    """Flatten a tree into a list of all its nodes (root + every descendant), via iterative DFS."""
    out, stack = [], [tree]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def mutate(tree: Node, sampler: FunctionSampler, rng) -> Node:
    """Return a mutated copy: perturb a constant, swap an operator of the
    same arity, or replace a random subtree with a fresh random one."""
    tree = copy.deepcopy(tree)          # never mutate the original in place
    node = rng.choice(_all_nodes(tree)) # pick one random node in the (copied) tree to mutate
    move = rng.random()                 # single dice roll to decide which kind of mutation to apply
    
    if move < 0.4 and node.op == "const":
        # Constant perturbation: nudge the value with small Gaussian noise
        node.value += rng.normal(0, 0.5)
    elif move < 0.7 and node.op in UNARY:
        # Operator swap: replace with another random unary op (arity-preserving, keeps 1 child)
        node.op = rng.choice(list(UNARY))
    elif move < 0.7 and node.op in BINARY:
        # Operator swap: replace with another random binary op (arity-preserving, keeps 2 children)
        node.op = rng.choice(list(BINARY))
    else:  # replace subtree
        # Fallback (also used when e.g. a non-const leaf falls outside the above branches):
        # grow a brand-new random subtree and graft it in place of the chosen node
        new = sampler._grow(depth=sampler.max_depth - 2)
        node.op, node.children, node.value = new.op, new.children, new.value
        
    return tree


def fit_constants(tree: Node, x: np.ndarray, y: np.ndarray):
    """Refine all constants in `tree` plus an outer affine a*g(x)+b by
    least squares. Returns (a, b, train_mse); mutates tree's constants."""
    consts = [n for n in _all_nodes(tree) if n.op == "const"]  # all constant leaves to optimize

    def residual(p):
        # p = [a, b, const_1, const_2, ...]: unpack and write constants back into the tree
        for n, v in zip(consts, p[2:]):
            n.value = v
            
        g = tree.evaluate(x)
        g = np.where(np.isfinite(g), g, 1e6)  # replace NaN/inf with a large finite penalty so the optimizer can still work
        
        return p[0] * g + p[1] - y  # residual of outer affine fit: a*g(x) + b - y

    p0 = np.array([1.0, 0.0] + [n.value for n in consts])  # initial guess: identity affine + current constants
    
    try:
        res = least_squares(residual, p0, max_nfev=200)  # nonlinear least-squares optimization (bounded # of evals)
        a, b = res.x[0], res.x[1]
        
        for n, v in zip(consts, res.x[2:]):
            n.value = v  # write the optimized constants back into the tree permanently
            
        return a, b, float(np.mean(res.fun ** 2))  # final affine coeffs + resulting train MSE
    except Exception:
        # Optimization failed (e.g. bad tree/numerical issues): fall back to identity affine, infinite loss
        return 1.0, 0.0, np.inf


def _xi(g_vals: np.ndarray, y: np.ndarray) -> float:
    """Chatterjee's xi(g(X), y); -inf for degenerate candidates."""
    if not np.all(np.isfinite(g_vals)) or np.std(g_vals) < 1e-12:
        # Guard against NaN/inf or near-constant candidates, which would make the
        # dependence measure meaningless or undefined
        return -np.inf
    
    return chatterjeexi(g_vals, y).statistic  # Chatterjee's rank-based correlation coefficient (measures dependence, not just linear correlation)


def _affine_mse(g_vals: np.ndarray, y: np.ndarray) -> float:
    """MSE of the closed-form best affine fit y ~ a*g + b (O(n))."""
    A = np.column_stack([g_vals, np.ones_like(g_vals)])  # design matrix: [g(x), 1] for a linear model a*g + b
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)          # closed-form least-squares solve (cheap, no iterative optimizer)
    return float(np.mean((A @ coef - y) ** 2))            # resulting mean squared error


# ---------------------------------------------------------------------------
# A) Chatterjee-xi-guided search
# ---------------------------------------------------------------------------

class XiGuidedSR(Regressor):
    """Hill-climbing over random expression trees, guided by xi + affine fit.

    Each roll (mutation of the current best, or a fresh random tree) is
    first screened with Chatterjee's xi: candidates whose xi falls clearly
    below the best xi seen are rejected before any fitting happens.
    This is done by comparing the xi between g(x) and y, where g is the candidate tree.
    There we do not look at the correctness of g(x) yet, but instead look whether
    the candidate g(x) is a function of y, where in a perfect solution it would be a
    linear diagonal line.
    Survivors are ranked by a closed-form affine-fit MSE:
        - find linear coefficients a, b such that y ~ a*g(x) + b
    Why xi cannot be the *only* guide in 1-D: xi(g(X), y) ~ 1 for ANY
    injective g of x (y = f(x) is trivially a function of g(X) then), so
    on noiseless 1-D data it saturates and stops discriminating between
    structures. It still prunes non-injective wrong shapes and stays
    informative under noise.
    """

    def __init__(self, n_iter=20000, top_k=5, p_restart=0.15,
                 xi_frac=0.9, parsimony=0.01, max_depth=4, seed=0):
        self.n_iter, self.top_k, self.p_restart = n_iter, top_k, p_restart  # search budget, # of finalists, restart chance
        self.xi_frac = xi_frac  # prune if xi < xi_frac * best_xi_seen
        self.parsimony = parsimony  # score = mse * (1 + parsimony * tree size)
        self.sampler = FunctionSampler(max_depth=max_depth, seed=seed)  # used both for the initial tree and for fresh restarts/subtree replacement
        self.rng = np.random.default_rng(seed)
        self.tree, self.a, self.b = None, 1.0, 0.0  # will hold the final chosen structure + outer affine fit

    def fit(self, x, y):
        def penalized(m, tree):        # complexity pressure against stowaway terms
            # Inflate raw MSE by a factor growing with tree size, so equally-fitting
            # but simpler trees are preferred (Occam's razor / anti-overfitting)
            return m * (1.0 + self.parsimony * tree.size())

        # --- initialize with one random tree as the current "best" ---
        best_tree = self.sampler._grow(0)
        g = best_tree.evaluate(x)
        best_xi = _xi(g, y) # Note we are calculating xi on g(x) and y, not x and y
        best_score = penalized(_affine_mse(g, y), best_tree)
        pool = [(best_score, best_tree)]  # keeps every tree that ever improved on best_score

        for _ in range(self.n_iter):
            if self.rng.random() < self.p_restart:
                cand = self.sampler._grow(0)          # fresh roll
                # occasionally escape local optima by sampling a wholly new random tree
            else:
                cand = mutate(best_tree, self.sampler, self.rng)
                # normally explore locally by mutating the current best

            g = cand.evaluate(x)
            xi = _xi(g, y)
            if xi < self.xi_frac * best_xi:           # xi prunes the roll
                # Cheap screening: skip full affine-MSE scoring if this candidate's
                # dependence on y is clearly worse than what's already been seen
                continue
            
            best_xi = max(best_xi, xi)  # track the best xi seen so far, to keep raising the bar
            score = penalized(_affine_mse(g, y), cand)  # candidate survived screening -> compute real (cheap, closed-form) fit quality
            
            if score < best_score:                    # decide how to continue
                # New best found: adopt it as the hill-climbing anchor and remember it
                best_tree, best_score = cand, score
            
            pool.append((score, cand))

        # full constant fitting only for the top-k structures
        pool.sort(key=lambda t: t[0])
        results = []
        
        for _, tree in pool[: self.top_k]:
            # Expensive nonlinear constant optimization is deferred to just the
            # handful of best structures found, not run on every candidate
            tree = copy.deepcopy(tree)
            a, b, m = fit_constants(tree, x, y)
            results.append((penalized(m, tree), a, b, tree))
            
        # pick the overall winner after constants have been properly refit
        _, self.a, self.b, self.tree = min(results, key=lambda t: t[0])
        return self

    def predict(self, x):
        # apply the fitted outer affine transform to the tree's raw output
        return self.a * self.tree.evaluate(x) + self.b

    def get_expression(self):
        import sympy as sp
        expr = sp.simplify(self.a * self.tree.to_sympy() + self.b)
        # 4 significant digits: small terms stay visible
        return str(sp.N(expr, 4))


# ---------------------------------------------------------------------------
# B) Conventional G(enetic)P(rogramming) symbolic regression (gplearn)
# ---------------------------------------------------------------------------

class GplearnSR(Regressor):
    """Classic genetic-programming SR via the gplearn library.
    Conceptually it works like biological evolution applied to expression trees:
        - Population: start with population_size (here 2000) completely random
        expression trees ("programs")
        - Fitness: each program is evaluated against the data (MSE)
        - Selection: choose better-fitting as "parents" for the next generation.
        - Breeding operators: new programs are created from parents using:
            - Crossover (p_crossover=0.7): two parent trees, cut random
            subtree from each, and swap them
            - Subtree mutation (p_subtree_mutation=0.1): take parent, replace
            a random subtree with a new random one
            - Hoist mutation (p_hoist_mutation=0.05): replace with fragment of itself
            - Point mutation (p_point_mutation=0.1): tweak a single operator
            or constant in place
        - Repeat: process runs for generations (here 30), each time population as a
        whole tends to get fitter, since good "genes"
        (useful subexpressions) survive and recombine while bad ones die out.
    """

    def __init__(self, generations=30, population_size=2000, seed=0):
        from gplearn.genetic import SymbolicRegressor
        from gplearn.functions import make_function
        # gplearn has no built-in exp; add a protected one so the operator
        # vocabulary matches the other methods (fairness on exp targets)
        exp_fn = make_function(function=lambda a: np.exp(np.clip(a, -20, 20)),
                               name="exp", arity=1)
        # "protected" here means clipped, like the custom Node UNARY dict, so
        # gplearn's exp can't overflow/produce inf during evolution — matching
        # how the hand-rolled sampler/evaluator guards against blow-ups
        self.est = SymbolicRegressor(
            population_size=population_size, generations=generations,
            # operator vocabulary deliberately mirrors UNARY/BINARY from the
            # custom Node-based approach, for an apples-to-apples comparison
            function_set=("add", "sub", "mul", "div",
                          "sin", "cos", "sqrt", "log", exp_fn),
            parsimony_coefficient=0.001,    # complexity penalty, gplearn's equivalent of the custom `parsimony` term
            p_crossover=0.7,                # probability of combining two parent programs (swap subtrees)
            p_subtree_mutation=0.1,         # probability of replacing a random subtree with a new random one
            p_hoist_mutation=0.05,          # probability of replacing a program with one of its own subtrees (anti-bloat)
            p_point_mutation=0.1,           # probability of randomly replacing a single node (operator/terminal) in place
            random_state=seed, verbose=0)

    def fit(self, x, y):
        # gplearn expects a 2D feature matrix (n_samples, n_features);
        # reshape the 1-D x into a single-column matrix
        self.est.fit(x.reshape(-1, 1), y)
        return self

    def predict(self, x):
        return self.est.predict(x.reshape(-1, 1))

    def get_expression(self):
        return str(self.est._program)  # gplearn's prefix-style notation


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datagen import generate_dataset, mse

    hero = Node("add", [
        Node("mul", [Node("const", value=0.5),
                     Node("mul", [Node("x"), Node("sin", [Node("x")])])]),
        Node("mul", [Node("const", value=0.1), Node("square", [Node("x")])]),
    ])
    ds = generate_dataset(hero, noise_std=0.01, seed=0)

    for reg in (XiGuidedSR(seed=0), GplearnSR(seed=0)):
        reg.fit(ds.x_train, ds.y_train)
        err = mse(reg.predict(ds.x_test), ds.y_test)
        print(f"{type(reg).__name__:12s} MSE_test={err:10.4f}  "
              f"f_hat = {reg.get_expression()}")
