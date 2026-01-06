import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import SGD, Adam
from tqdm import tqdm

from scg import SCG


def rosenbrock(xy: torch.Tensor) -> torch.Tensor:
    """Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²"""
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def run_optimization_scg(xy_init, n_iter):
    """Run SCG optimization (using new API without closures)."""
    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = SCG([xy_t])

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init

    def loss_fn():
        return rosenbrock(xy_t)

    for i in tqdm(range(1, n_iter + 1), desc="SCG"):
        optimizer.step(loss_fn)
        path[i, :] = xy_t.detach().numpy()

    return path


def run_optimization_standard(xy_init, optimizer_class, n_iter, **kwargs):
    """Run standard optimizer (Adam, SGD, etc.)."""
    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = optimizer_class([xy_t], **kwargs)

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1), desc=optimizer_class.__name__):
        optimizer.zero_grad()
        loss = rosenbrock(xy_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
        optimizer.step()
        path[i, :] = xy_t.detach().numpy()

    return path


def plot_optimization_paths(
    paths,
    colors,
    names,
    figsize=(10, 10),
    x_lim=(-2, 1.5),
    y_lim=(-2, 1.5),
    n_points=300,
):
    """Plot the Rosenbrock function contours with optimization paths."""
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError("paths, colors, and names must have same length")

    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(torch.tensor(np.array([X, Y]))).numpy()

    minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")

    for path, color, name in zip(paths, colors, names):
        ax.plot(path[:, 0], path[:, 1], ".-", label=name, c=color)

    ax.legend(prop={"size": 16})
    ax.plot(*minimum, "rD", markersize=8)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    plt.tight_layout()
    plt.show()


def main():
    xy_init = (-1.95, -1.95)
    n_iter = 500

    print("=" * 60)
    print("Testing SCG on Rosenbrock function")
    print("=" * 60)

    path_scg = run_optimization_scg(xy_init, n_iter)
    path_adam = run_optimization_standard(xy_init, Adam, n_iter, lr=0.03)
    path_sgd = run_optimization_standard(xy_init, SGD, n_iter, lr=0.03)

    print("\nFinal positions:")
    print(f"  SCG:     {path_scg[-1]}")
    print(f"  Adam:    {path_adam[-1]}")
    print(f"  SGD:     {path_sgd[-1]}")
    print("  Optimal: (1.0, 1.0)")

    # Check convergence
    def dist_to_min(path):
        return np.sqrt((path[-1, 0] - 1) ** 2 + (path[-1, 1] - 1) ** 2)

    print("\nDistance to minimum:")
    print(f"  SCG:   {dist_to_min(path_scg):.6f}")
    print(f"  Adam:  {dist_to_min(path_adam):.6f}")
    print(f"  SGD:   {dist_to_min(path_sgd):.6f}")

    # Plot the results
    plot_optimization_paths(
        paths=[path_sgd, path_adam, path_scg],
        colors=["g", "b", "k"],
        names=["SGD", "Adam", "SCG"],
    )


if __name__ == "__main__":
    main()
