import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import BasisLinearModel, NNXBasisLinearModel
from .plotting import compare_linear_models, plot_fit, plot_basis
from .training import train


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data_generating_model = BasisLinearModel(
        weights=np_rng.normal(loc=0, scale=1, size=(settings.data.num_basis)),
        mu=np_rng.uniform(low=0.5, high=1.5, size=(settings.data.num_basis)),
        sigma=np_rng.normal(loc=0, scale=1, size=(settings.data.num_basis)),
        bias=0,
    )
    log.debug("Data generating model", model=data_generating_model)

    data = Data(
        model=data_generating_model,
        rng=np_rng,
        num_features=settings.data.num_features,
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
    )

    model = NNXBasisLinearModel(
        rngs=nnx.Rngs(params=model_key),
        num_features=settings.data.num_features,
        num_basis=settings.data.num_basis,
        sigma=settings.data.sigma_noise,
    )
    log.debug("Initial model", model=model.model)

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train(model, optimizer, data, settings.training, np_rng)

    log.debug("Trained model", model=model.model)

    compare_linear_models(data.model, model.model)

    if settings.data.num_features == 1:
        plot_fit(model, data, settings.plotting)
        plot_basis(model, settings.plotting, settings.data)

    else:
        log.info("Skipping plotting for multi-feature models.")
