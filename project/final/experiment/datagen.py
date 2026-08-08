"""Random symbolic function sampling + train/test data generation.

  * sample random 1-D symbolic functions f(x) as expression trees
    ("roll dice over operators")
  * generate datasets: noisy train points inside a domain, clean
    out-of-domain test points for extrapolation scoring

Design note: functions are represented as expression *trees*. The
simplest form of a graph, where nodes are symbols (operators, the
variable x, constants) and edges connect operators to their operands.
DAGs with shared subexpressions (Kahlmeyer et al., 2025) only pay off
for the *search* side later, not for sampling, so we keep trees here.

Numerical safety guards (protected log/sqrt/div, clipped exp) follow
the data-generation practice of Scholl et al. (ParFam, 2025).
"""

from dataclasses import dataclass, field
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Operators: name -> (numpy implementation, sympy implementation)
# All risky operators are domain-guarded so sampled functions never crash.
# ---------------------------------------------------------------------------

EPS = 1e-9

UNARY = {
    "sin":    (np.sin,                         sp.sin),
    "cos":    (np.cos,                         sp.cos),
    "exp":    (lambda a: np.exp(np.clip(a, -20, 20)), sp.exp),
    "log":    (lambda a: np.log(np.abs(a) + EPS), lambda a: sp.log(sp.Abs(a))),
    "sqrt":   (lambda a: np.sqrt(np.abs(a)), lambda a: sp.sqrt(sp.Abs(a))),
    "square": (np.square, lambda a: a**2),
    "neg":    (np.negative, lambda a: -a),
}

BINARY = {
    "add": (np.add, lambda a, b: a + b),
    "sub": (np.subtract, lambda a, b: a - b),
    "mul": (np.multiply, lambda a, b: a * b),
    "div": (lambda a, b: a / np.where(np.abs(b) < EPS, EPS, b), lambda a, b: a / b),
}


@dataclass
class Node:
    """A node in the expression tree. Leaves: op='x' or op='const'."""
    op: str                                         # operation name: "x", "const", a key in UNARY, or a key in BINARY
    children: list = field(default_factory=list)    # child Nodes (0 for leaves, 1 for unary ops, 2 for binary ops)
    value: float = None                             # the constant's numeric value, only set when op == "const"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Numerically evaluate this subtree at the given x values (vectorized over a numpy array)."""
        if self.op == "x":
            # Leaf node: identity function, just return the input array
            return x
        
        if self.op == "const":
            # Leaf node: broadcast the stored constant to the shape of x
            return np.full_like(x, self.value, dtype=float)
        
        if self.op in UNARY:
            # Unary op: recursively evaluate the single child, then apply the numpy function
            # (index 0 of the UNARY tuple is the numpy implementation)
            return UNARY[self.op][0](self.children[0].evaluate(x))
        
        # Binary op: recursively evaluate both children, then apply the numpy function
        # (index 0 of the BINARY tuple is the numpy implementation)
        return BINARY[self.op][0](*(c.evaluate(x) for c in self.children))

    def to_sympy(self):
        """Convert this subtree into a symbolic sympy expression (for display/simplification, not evaluation)."""
        x = sp.Symbol("x", real=True)  # symbolic variable matching the leaf op='x'
        
        if self.op == "x":
            # Leaf node: return the sympy symbol
            return x
        
        if self.op == "const":
            # Leaf node: return the constant as a sympy Float
            return sp.Float(self.value)
        
        if self.op in UNARY:
            # Unary op: recursively convert the child, then apply the sympy version of the function
            # (index 1 of the UNARY tuple is the sympy implementation, kept separate from numpy
            # since sympy has no direct equivalents for things like np.clip)
            return UNARY[self.op][1](self.children[0].to_sympy())
        
        # Binary op: recursively convert both children, then apply the sympy version of the function
        return BINARY[self.op][1](*(c.to_sympy() for c in self.children))

    def size(self) -> int:
        """Count total number of nodes in this subtree (used e.g. for complexity penalties)."""
        return 1 + sum(c.size() for c in self.children)

    def __str__(self):
        # Human-readable form: convert to sympy and simplify algebraically before printing
        return str(sp.simplify(self.to_sympy()))


class FunctionSampler:
    """Samples random symbolic functions f(x) by rolling dice over operators."""

    def __init__(self, max_depth=4, p_terminal=0.35, p_unary=0.35,
                 const_range=(-2.0, 2.0), y_bound=1000.0, seed=None):
        self.max_depth = max_depth      # cap on tree depth, prevents infinite/huge recursion
        self.p_terminal = p_terminal    # probability of stopping and generating a leaf at each node
        self.p_unary = p_unary          # probability (after not stopping) of picking a unary op over a binary op
        self.const_range = const_range  # (min, max) range for sampling leaf constants
        self.y_bound = y_bound          # reject functions that explode on the domain
        self.rng = np.random.default_rng(seed)  # dedicated RNG so sampling is reproducible via seed

    def _grow(self, depth: int) -> Node:
        """Recursively build a random expression tree, biasing toward leaves as depth increases."""
        r = self.rng.random()  # single dice roll used to decide what kind of node to create

        if depth >= self.max_depth or r < self.p_terminal:
            # Force a leaf once max depth is reached, or by chance (p_terminal)
            # leaf: variable (more likely) or constant
            if self.rng.random() < 0.7:
                return Node("x")  # 70% chance: variable leaf
            return Node("const", value=self.rng.uniform(*self.const_range))  # 30% chance: random constant leaf

        if r < self.p_terminal + self.p_unary:
            # Next slice of probability: pick a random unary op and grow one child
            op = self.rng.choice(list(UNARY))
            return Node(op, [self._grow(depth + 1)])

        # Remaining probability: pick a random binary op and grow two children
        op = self.rng.choice(list(BINARY))
        return Node(op, [self._grow(depth + 1), self._grow(depth + 1)])

    def _is_valid(self, f: Node, domain) -> bool:
        """Reject functions that are NaN/inf, exploding, or constant on the domain."""
        x = np.linspace(*domain, 200)  # sample points across the domain to sanity-check f
        y = f.evaluate(x)
        
        return (np.all(np.isfinite(y))                  # no NaN/inf values anywhere
                and np.max(np.abs(y)) < self.y_bound    # doesn't blow up beyond the allowed bound
                and np.std(y) > 1e-6                    # not (numerically) constant
                # actually depends on x after simplification
                and "x" in str(f))                      # after sympy simplification, x must still appear (e.g. rules out x - x)

    def sample(self, domain=(-5.0, 5.0), max_tries=100) -> Node:
        """Sample one valid random function; retries until the sanity filter passes."""
        for _ in range(max_tries):
            f = self._grow(depth=0)       # generate a fresh random tree from the root
            if self._is_valid(f, domain):
                return f
            
        raise RuntimeError("no valid function found; relax sampler settings")  # gave up after max_tries attempts


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray   # noisy labels used for fitting
    x_test: np.ndarray
    y_test: np.ndarray    # CLEAN ground truth, out-of-domain (for MSE_test)
    expression: str       # ground-truth formula (for recovery checks / plots)
    train_domain: tuple
    test_band: float
    noise_std: float
    seed: int


def generate_dataset(f: Node, n_train=200, n_test=200, train_domain=(-5.0, 5.0),
                     test_band=3.0, noise_std=0.0, seed=0) -> Dataset:
    """Train points inside `train_domain` (noisy), test points in the bands
    of width `test_band` on both sides outside it (clean ground truth)."""
    rng = np.random.default_rng(seed)
    lo, hi = train_domain

    x_train = rng.uniform(lo, hi, n_train)
    y_train = f.evaluate(x_train) + rng.normal(0.0, noise_std, n_train)

    # out-of-domain test points, half on each side
    left = rng.uniform(lo - test_band, lo, n_test // 2)
    right = rng.uniform(hi, hi + test_band, n_test - n_test // 2)
    x_test = np.concatenate([left, right])
    y_test = f.evaluate(x_test)  # clean: score against the true function

    return Dataset(x_train, y_train, x_test, y_test, str(f),
                   train_domain, test_band, noise_std, seed)


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Extrapolation metric: MSE on the (clean) out-of-domain test set."""
    return float(np.mean((y_pred - y_true) ** 2))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sampler = FunctionSampler(seed=42)
    for i in range(5):
        f = sampler.sample()
        ds = generate_dataset(f, noise_std=0.01, seed=i)
        print(f"f_{i}(x) = {ds.expression}   (tree size {f.size()}, "
              f"{len(ds.x_train)} train / {len(ds.x_test)} test points)")

    # the slides' hero example, built by hand from the same Node class
    hero = Node("add", [
        Node("mul", [Node("const", value=0.5),
                     Node("mul", [Node("x"), Node("sin", [Node("x")])])]),
        Node("mul", [Node("const", value=0.1), Node("square", [Node("x")])]),
    ])
    ds = generate_dataset(hero, noise_std=0.01)
    print(f"\nhero example: f(x) = {ds.expression}")
    print(f"train x in [{ds.x_train.min():.2f}, {ds.x_train.max():.2f}], "
          f"test x in [{ds.x_test.min():.2f}, {ds.x_test.max():.2f}]")
