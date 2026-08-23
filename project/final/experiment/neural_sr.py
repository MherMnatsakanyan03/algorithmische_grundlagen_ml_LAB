"""C) NeuralSR -- a neural network that predicts a *formula* from data.

we feed the network the y-values interpolated onto a fixed
64-point x-grid over the training interval, normalized.

Output: predicts the formula as a token sequence in prefix notation,
where every numeric constant is the placeholder token 'C' (skeleton).
After decoding, the constants (plus an outer affine a*g(x)+b) are
fitted with least squares on the actual data.
architecture is a small MLP encoder + GRU decoder instead of a
transformer, which is plenty for 1-D.
"""

import numpy as np
import torch
import torch.nn as nn

from datagen import Node, FunctionSampler, UNARY, BINARY
from regressors import Regressor, fit_constants

# vocabulary: specials + terminals + operators
# <pad>: padding for batching variable-length sequences
# <sos>/<eos>: start/end-of-sequence markers for the decoder
# x: the input variable terminal
# C: placeholder for ANY numeric constant (actual values are fit later, not predicted by the net)
# followed by all unary op names (sin, cos, exp, ...) and binary op names (add, sub, ...)
TOKENS = ["<pad>", "<sos>", "<eos>", "x", "C"] + list(UNARY) + list(BINARY)
TOK2ID = {t: i for i, t in enumerate(TOKENS)}  # token string -> integer id, for embedding lookup / loss computation
N_GRID, MAX_LEN = 64, 24  # N_GRID: fixed number of points the (x,y) data is resampled to for network input
                          # MAX_LEN: max number of tokens allowed in a predicted formula (skeleton) sequence


# ------------------------- tree <-> token sequence -------------------------

def to_tokens(node: Node) -> list:
    """Prefix notation; constants become the placeholder 'C'."""
    # Recursively serialize the tree into a flat list of tokens: op first,
    # then all children's tokens in order (prefix / Polish notation),
    # so the tree structure can be recovered unambiguously from arities alone
    if node.op == "const":
        return ["C"]  # any constant leaf collapses to the same placeholder token (the "skeleton" abstraction)
    return [node.op if node.op != "x" else "x"] + \
        [t for c in node.children for t in to_tokens(c)]


def parse_tokens(tokens: list) -> Node:
    """Inverse of to_tokens ('C' -> const node with value 1.0)."""
    it = iter(tokens)  # consume tokens one at a time as the tree is rebuilt

    def build():
        # Recursive-descent parser: reads one token, and if it's an operator,
        # recursively builds exactly as many children as that operator's arity requires
        t = next(it)
        
        if t == "x":
            return Node("x")
        if t == "C":
            return Node("const", value=1.0)  # dummy placeholder value; real constants are fit afterward via least squares
        if t in UNARY:
            return Node(t, [build()])   # arity 1: consume one child subtree
        if t in BINARY:
            return Node(t, [build(), build()])  # arity 2: consume two child subtrees, in order
        raise ValueError(f"bad token {t}")

    return build()


def encode_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate (x, y) onto a fixed grid and normalize -> model input."""
    order = np.argsort(x)  # np.interp requires x-coordinates to be sorted ascending
    grid = np.linspace(x.min(), x.max(), N_GRID)  # fixed-size grid so the network always sees a constant-length input,
                                                    # regardless of how many (possibly irregular) data points were given
    yg = np.interp(grid, x[order], y[order])  # linear interpolation of y onto the regular grid
    return ((yg - yg.mean()) / (yg.std() + 1e-9)).astype(np.float32)
    # standardize (zero mean, unit std) so the network is invariant to the
    # absolute scale/offset of y -- that info is recovered later via the
    # outer affine fit (a*g(x)+b), same role EPS-style epsilon avoids div-by-zero


# --------------------------------- model -----------------------------------

class _Seq2Seq(nn.Module):
    """Encodes a resampled y-signal into a hidden vector, then autoregressively
    decodes a token sequence (the formula skeleton) conditioned on it.
    (kinda like LLMs, but much smaller and simpler: no attention, no transformers, just a small MLP + GRU.)
    """

    def __init__(self, hidden=128, emb=64):
        super().__init__()
        # "Encoder": squashes the 64 normalized y-values into one hidden vector.
        # Just an MLP; no attention, no notion of individual (x,y) points,
        # it treats the whole grid as one flat feature vector.
        self.encoder = nn.Sequential(
            nn.Linear(N_GRID, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.embed = nn.Embedding(len(TOKENS), emb)  # token id -> learned vector, for feeding into the GRU
        self.gru = nn.GRU(emb, hidden, batch_first=True)  # autoregressive decoder: one token in, one hidden state out
        self.out = nn.Linear(hidden, len(TOKENS))  # projects each GRU hidden state to a distribution over the vocabulary

    def forward(self, y_grid, token_ids): # ("teacher forcing")
        # Training-time forward pass: the *true* token sequence (shifted by one)
        # is fed in directly, rather than the model's own previous predictions
        h0 = self.encoder(y_grid).unsqueeze(0)  # (batch, hidden) -> (1, batch, hidden): GRU's initial hidden state (num_layers=1)
        o, _ = self.gru(self.embed(token_ids), h0)  # run the whole target sequence through in one shot
        return self.out(o)  # logits at every position, compared against the next tokens via cross-entropy

    @torch.no_grad()
    def decode(self, y_grid): # greedy decoding
        # Inference-time: no ground truth available, so the model must feed its
        # own predictions back in as the next input, one token at a time
        h = self.encoder(y_grid.unsqueeze(0)).unsqueeze(0)  # encode this one example -> initial hidden state
        tok = torch.tensor([[TOK2ID["<sos>"]]], device=y_grid.device)  # start the sequence with <sos>
        out = []
        for _ in range(MAX_LEN):
            o, h = self.gru(self.embed(tok), h)  # one GRU step; hidden state carries all prior context forward
            tok = self.out(o[:, -1]).argmax(-1, keepdim=True)  # greedy: always pick the single highest-probability next token
            # Herem self.out is the probability distribution over the vocabulary for the next token, given the current hidden state
            t = TOKENS[tok.item()]
            if t == "<eos>":
                break  # stop once the model predicts end-of-sequence
            out.append(t)
        return out  # the predicted token skeleton, without <sos>/<eos>


# ------------------------------- regressor ---------------------------------

class NeuralSR(Regressor):
    """Pretrained net predicts a skeleton; constants fitted per dataset."""

    def __init__(self, weights_path=None, seed=0, device=None):
        torch.manual_seed(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _Seq2Seq().to(self.device)
        if weights_path:
            # load an already-pretrained model instead of training from scratch
            # (pretraining is meant to happen once, offline, not per dataset)
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device))
        self.tree, self.a, self.b = None, 1.0, 0.0

    # ---- one-time pretraining on the function generator ----
    def pretrain(self, n_functions=20000, batch_size=64, epochs=3,
                 lr=1e-3, domain=(-5.0, 5.0), seed=0, save_path=None):
        # Builds a synthetic supervised dataset: (resampled y-signal) -> (token skeleton),
        # using the SAME random function sampler as the other methods, so the net
        # learns the same "distribution of plausible formulas" they search over
        sampler = FunctionSampler(max_depth=3, seed=seed)  # shallower than default: keeps token sequences short enough for MAX_LEN
        rng = np.random.default_rng(seed)
        grids, targets = [], []
        while len(grids) < n_functions:
            f = sampler.sample(domain=domain)
            toks = to_tokens(f)
            if len(toks) + 2 > MAX_LEN:   # +2 for <sos>/<eos>; skip if it wouldn't fit
                continue
            x = rng.uniform(*domain, 200)   # fresh random x-samples for this synthetic function
            y = f.evaluate(x)
            grids.append(encode_points(x, y))  # network input: normalized y on the fixed grid
            ids = [TOK2ID["<sos>"]] + [TOK2ID[t]
                                       for t in toks] + [TOK2ID["<eos>"]]
            targets.append(ids + [TOK2ID["<pad>"]] * (MAX_LEN + 2 - len(ids)))
            # pad every target to the same fixed length so they can be batched into one tensor

        G = torch.tensor(np.stack(grids))
        T = torch.tensor(targets)
        print(f"pretraining on {self.device} ({len(G)} functions)")
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss(ignore_index=TOK2ID["<pad>"])  # padding tokens don't contribute to the loss
        self.model.train()
        for ep in range(epochs):
            perm, total = torch.randperm(len(G)), 0.0  # shuffle each epoch
            for i in range(0, len(G), batch_size):
                idx = perm[i:i + batch_size]
                g = G[idx].to(self.device)
                t = T[idx].to(self.device)
                # teacher forcing: feed tokens[:-1] as input, train to predict tokens[1:]
                # (classic "shifted sequence" next-token training used in seq2seq/LM training)
                logits = self.model(g, t[:, :-1])
                loss = lossf(logits.reshape(-1, len(TOKENS)),
                             t[:, 1:].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"epoch {ep + 1}: loss {total / (len(G) / batch_size):.3f}")
        self.model.eval()
        if save_path:
            torch.save(self.model.state_dict(), save_path)  # persist weights so future runs can skip pretraining
        return self

    # ---- per-dataset usage ----
    def fit(self, x, y):
        # At use-time: encode the REAL dataset, greedily decode a predicted skeleton,
        # then recover actual numeric constants via the same least-squares routine
        # used by XiGuidedSR (skeleton search decoupled from constant values)
        grid = torch.tensor(encode_points(x, y), device=self.device)
        try:
            self.tree = parse_tokens(self.model.decode(grid))
        except (ValueError, StopIteration):       # invalid decode -> fallback
            # model produced a malformed/incomplete token sequence
            # (e.g. an operator with missing children) -- fall back to a trivial model
            self.tree = Node("x")
        self.a, self.b, _ = fit_constants(self.tree, x, y)
        return self

    def predict(self, x):
        return self.a * self.tree.evaluate(x) + self.b

    def get_expression(self):
        import sympy as sp
        expr = sp.simplify(self.a * self.tree.to_sympy() + self.b)
        return str(sp.N(expr, 4))


if __name__ == "__main__":
    from datagen import generate_dataset, mse

    reg = NeuralSR()
    reg.pretrain(n_functions=2000, epochs=2,
                 save_path="neural_sr.pt")  # smoke run

    f = FunctionSampler(seed=7).sample()
    ds = generate_dataset(f, noise_std=0.01, seed=1)
    reg.fit(ds.x_train, ds.y_train)
    print(f"true f  = {ds.expression}")
    print(f"f_hat   = {reg.get_expression()}")
    print(f"MSE_test= {mse(reg.predict(ds.x_test), ds.y_test):.4f}")
