"""Figure for the report: the hero function and the three regressed
functions, with the extrapolation region visible.

Run:  python3 plot_hero.py       -> writes hero_plot.pdf (vector, for LaTeX)
Needs neural_sr.pt (the pretrained weights) in the same folder.
"""

import numpy as np
import matplotlib.pyplot as plt

from datagen import Node, generate_dataset
from regressors import XiGuidedSR, GplearnSR
from neural_sr import NeuralSR

# same dataset as in the experiments (seed 0, noise 0.01, band 3)
hero = Node("add", [
    Node("mul", [Node("const", value=0.5),
                 Node("mul", [Node("x"), Node("sin", [Node("x")])])]),
    Node("mul", [Node("const", value=0.1), Node("square", [Node("x")])]),
])
ds = generate_dataset(hero, noise_std=0.01, seed=0)

regressors = {
    "A: $\\xi$-guided": XiGuidedSR(seed=0),
    "B: gplearn":       GplearnSR(seed=0),
    "C: neural":        NeuralSR(weights_path="neural_sr.pt"),
}

xs = np.linspace(-8, 8, 800)
fig, ax = plt.subplots(figsize=(6.0, 3.4))

# training interval shaded, everything outside is extrapolation
ax.axvspan(-5, 5, color="0.92", zorder=0, label="training interval")
ax.plot(xs, hero.evaluate(xs), "k-", lw=2, label="true $f(x)$", zorder=3)
ax.plot(ds.x_train, ds.y_train, ".", color="0.55", ms=3,
        label="train points", zorder=2)

styles = ["--", "-.", ":"]
for (name, reg), style in zip(regressors.items(), styles):
    print(f"fitting {name} ...")
    reg.fit(ds.x_train, ds.y_train)
    ax.plot(xs, reg.predict(xs), style, lw=1.8, label=name, zorder=4)

ax.set_xlim(-8, 8)
ax.set_ylim(-4, 12)          # clip exploding predictions to keep the plot readable
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend(loc="upper center", ncol=2, fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig("hero_plot.pdf")
print("wrote hero_plot.pdf")