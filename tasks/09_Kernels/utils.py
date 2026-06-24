from typing import Callable, Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes


def plot_decision_boundary(x: npt.NDArray[np.float64], y: npt.NDArray[np.bool_], predict: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], ax: Axes | None = None):
    '''
    Convenience function for plotting decision boundary of binary classificator.

    @Params:
        X... features (shape n_samples x feature_dim)
        y... labels (shape n_samples)
        predict... function for binary classification
        ax... matplotlib axis on which to plot
    '''
    if ax is None:
        ax = plt.gca()

    # make these smaller to increase the resolution
    x_1 = x[:, -2]
    x_2 = x[:, -1]
    mesh_distance_x_1, mesh_distance_x_2 = 0.01, 0.01
    # generate grids + labels
    mesh_x_1, mesh_x_2 = np.mgrid[
        slice(np.min(x_1), np.max(x_1) + mesh_distance_x_1, mesh_distance_x_1),
        slice(np.min(x_2), np.max(x_2) + mesh_distance_x_2, mesh_distance_x_2),
    ]
    n_mesh_points = np.prod(mesh_x_1.shape)
    mesh_points = np.stack([np.ones(n_mesh_points), mesh_x_1.flatten(), mesh_x_2.flatten()]).T
    if x.shape[1] == 2:
        mesh_points = mesh_points[:, 1:]
    mesh_labels = predict(mesh_points).reshape(mesh_x_1.shape)

    # plot points + areas
    cmap = plt.get_cmap("bwr")
    ax.contourf(
        mesh_x_1,
        mesh_x_2,
        mesh_labels,
        cmap=cmap,
        alpha=0.4,
        vmin=0,
        vmax=1
    )
    # plt.colorbar()
    ax.scatter(x_1, x_2, c=y, cmap="bwr", s=8)


def sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Sigmoid function.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Array of any dimension with scalar entries.

    Returns
    -------
    npt.NDArray[np.float64]
        Array of the same shape as the input with the sigmoid function applied element-wise.
    """

    return 1 / (1 + np.exp(-x))


class LogisticRegressor():
    def __init__(self, compute_loss_and_gradient: Callable[[npt.NDArray[np.float64], npt.NDArray[np.int8], npt.NDArray[np.float64]], tuple[float, npt.NDArray[np.float64]]], learn_rate: float = 1e-2, max_iterations: int = 1000, epsilon: float = 1e-5):
        """Regressor for binary classification using logistic regression. Fits using gradient descent.

        Parameters
        ----------
        compute_loss_and_gradient : Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int8]], tuple[float, npt.NDArray[np.float64]]]
            Function that computes the loss and gradient for the given the features, labels (in {-1, 1}), and parameters (in that order).
        learn_rate : float, optional
            Learning rate, sets step size for descent, by default 1e-2.
        max_iterations : int, optional
            Maximum number of descent steps, by default 1000.
        epsilon : float, optional
            Descent stops early if the loss did not change more than this, by default 1e-5.
        """

        self.compute_loss_and_gradient = compute_loss_and_gradient
        self.learn_rate = learn_rate
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.theta = np.empty(0, dtype=np.float64)
        self.accuracies = []
        self.losses = []

    def probability(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predicts probabilities of y=1 given features x and learned parameters theta.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Matrix with datapoints as rows (m x n).

        Returns
        -------
        npt.NDArray[np.float64]
            Array of probabilities (m), entries in [0, 1].
        """

        return sigmoid(x @ self.theta)

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.int8]:
        """Predicts labels y given features x and learned parameters theta.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Matrix with datapoints as rows (m x n).

        Returns
        -------
        npt.NDArray[np.int8]
            Array of predictions (m), entries in {-1, 1}.
        """

        return 2 * ((x @ self.theta) >= 0) - 1

    def fit(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.int8]) -> Self:
        """Gradient descent for binary crossentropy. Starts at a random parameter vector and tracks losses and accuracies along the iterations.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Matrix with datapoints as rows (m x n).
        y : npt.NDArray[np.int8]
            Vector of true labels (m), entries in {-1, 1}.
        """

        self.theta = np.random.rand(x.shape[1])
        for _ in range(self.max_iterations):
            loss, gradient = self.compute_loss_and_gradient(x, y, self.theta)
            self.theta -= self.learn_rate * gradient
            self.losses.append(loss)
            self.accuracies.append(np.mean(self.predict(x) == y))
            if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]) < self.epsilon:
                break
        return self
