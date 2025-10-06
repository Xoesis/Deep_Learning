import jax
import numpy as np
import optax
import structlog
from pathlib import Path
import orbax.checkpoint as ocp
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import Classifier
from .training import train, compute_accuracy


def main() -> None:
    """CLI entry point."""
    jax.debug.print("wth is going on")
    settings = load_settings()
    jax.debug.print("passed settings")
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = Data(
        rng=np_rng,
        train_val_split=settings.data.train_val_split,
        cifar10or100=settings.data.cifar10or100,
    )

    model = Classifier(
        rngs=nnx.Rngs(params=model_key),
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depth,
        layer_kernel_sizes=settings.model.kernel,
        stride=settings.model.stride,
        num_classes=settings.model.num_classes,
        l2pen=settings.data.l2pen,
        dropout_rate=settings.data.dropout_rate,
    )

    """Helps convergence by decreasing the learning rate over time"""
    schedule = optax.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=settings.training.num_iters,
    )

    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    train(model, optimizer, data, settings.training, np_rng)

    """Evaluate the accuracy of the training set"""
    val_accuracy = compute_accuracy(
        model, data, settings.training.batch_size, validation=True
    )

    ckpt_dir = ocp.test_utils.erase_and_create_empty("/tmp/cifar/")
    _, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "saved model", state)
    checkpointer.wait_until_finished()
    log.info("Saved model and tested on val set. ", Accuracy=val_accuracy)


def final_test():
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = Data(
        rng=np_rng,
        train_val_split=settings.data.train_val_split,
        cifar10or100=settings.data.cifar10or100,
    )

    model = Classifier(
        rngs=nnx.Rngs(params=model_key),
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depth,
        layer_kernel_sizes=settings.model.kernel,
        stride=settings.model.stride,
        num_classes=settings.mode.num_classes,
        l2pen=settings.data.l2pen,
        dropout_rate=settings.data.dropout_rate,
    )

    ckpt_dir = Path("/tmp/cifar/")
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    restored_state = checkpointer.restore(ckpt_dir / "saved model", state)
    model = nnx.merge(graphdef, restored_state)

    test_accuracy = compute_accuracy(
        model, data, settings.training.batch_size, validation=False
    )
    log.info("Final Test Set.", Accuracy=test_accuracy)
