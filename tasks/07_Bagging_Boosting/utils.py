import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
import seaborn as sns
import warnings


def plot_decision_boundary(x: npt.NDArray[np.float64], y: npt.NDArray[np.bool_], estimator, ax: Axes | None = None):
    '''
    Convenience function for plotting decision boundary of binary classificator.

    @Params:
        X... features (shape n_samples x feature_dim)
        y... labels (shape n_samples)
        estimator... instance of a class with .fit() and .predict() for binary classification
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
    mesh_labels = estimator.predict(mesh_points).reshape(mesh_x_1.shape)

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


def plot_density(x: npt.NDArray[np.float64], ax: Axes | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if ax is None:
            ax = plt.gca()
        sns.kdeplot(ax=ax, x=x[:, 0], y=x[:, 1], cmap="Reds", fill=True)
