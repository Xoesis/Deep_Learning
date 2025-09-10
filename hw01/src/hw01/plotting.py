import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import structlog

from .config import PlottingSettings, DataSettings
from .data import Data
from .model import BasisLinearModel, NNXBasisLinearModel

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def compare_linear_models(a: BasisLinearModel, b: BasisLinearModel):
    """Prints a comparison of two linear models."""
    log.info("Comparing models", true=a, estimated=b)
    print("w,    w_hat")
    for w_a, w_b in zip(a.weights, b.weights):
        print(f"{w_a:0.2f}, {w_b:0.2f}")

    print(f"{a.bias:0.2f}, {b.bias:0.2f}")


def plot_fit(
    model: NNXBasisLinearModel,
    data: Data,
    settings: PlottingSettings,
):
    """Plots the Basis-linear fit and saves it to a file."""
    log.info("Plotting fit")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("Basis-Linear fit")
    ax.set_xlabel("x")
    ax.set_ylim(np.amin(data.y) * 1.5, np.amax(data.y) * 1.5)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 100)
    xs = xs[:, np.newaxis]
    ax.plot(
        xs, np.squeeze(model(jnp.asarray(xs))), "-", np.squeeze(data.x), data.y, "o"
    )

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw01.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))


def plot_basis(
    model: NNXBasisLinearModel,
    settings: PlottingSettings,
    data: DataSettings,
):
    """Plots the Basis functions and saves it as a file."""
    log.info("Plotting Basis")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("Basis Functions")
    ax.set_xlabel("x")
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, 1.1)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    x = np.linspace(-1, 2, 100)

    params = model.model
    for i in range(data.num_basis):
        ax.plot(
            x, np.exp((-1 * ((x - params.mu[i]) ** 2)) / (params.sigma[i] ** 2)), "-"
        )

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw01_basis.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
