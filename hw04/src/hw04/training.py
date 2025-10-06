import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
import optax
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import Classifier

log = structlog.get_logger()


@nnx.jit
def train_step(
    model: Classifier, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
):
    """Performs a single training step."""

    def loss_fn(model: Classifier):
        func = model(x, True)
        ce_loss = jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(func, y)
        )
        l2_loss = model.l2_loss()
        return ce_loss + l2_loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # In-place update of model parameters
    return loss


def compute_accuracy(
    model: Classifier,
    data: Data,
    batch_size: int,
    validation: bool = True,
) -> float:
    if validation:
        x_np, y_np = data.get_val()
    else:
        x_np, y_np = data.get_test()
    x, y = (jnp.asarray(x_np[0:batch_size]), jnp.asarray(y_np[0:batch_size]))
    f = model(x, False)
    pred = jnp.argmax(f, axis=1)
    return round(jnp.mean(pred == y), 6)


def train(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        loss = train_step(model, optimizer, x, y)

        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()

    log.info("Training finished")
