from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class BasisLinearModel:
    """Represents a simple linear model."""

    weights: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    bias: float


class NNXBasisLinearModel(nnx.Module):
    """A Flax NNX module for a linear regression model."""

    def __init__(
        self, *, rngs: nnx.Rngs, num_features: int, num_basis: int, sigma: float
    ):
        self.num_features = num_features
        self.num_basis = num_basis
        key = rngs.params()
        self.sigma = nnx.Param(jax.random.normal(key, (1, self.num_basis)))
        self.mu = nnx.Param(jax.random.normal(key, (1, self.num_basis)))
        self.w = nnx.Param(jax.random.normal(key, (self.num_basis, 1)))
        self.b = nnx.Param(jnp.zeros((1, 1)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        x = x.reshape(-1, 1)
        phi = jnp.exp(-(((x - self.mu) / self.sigma) ** 2))
        return jnp.squeeze(phi @ self.w.value + self.b.value)

    @property
    def model(self) -> BasisLinearModel:
        """Returns the underlying simple linear model."""
        return BasisLinearModel(
            weights=np.array(self.w.value).reshape([self.num_basis]),
            mu=np.array(self.mu.value).reshape([self.num_basis]),
            sigma=np.array(self.sigma.value).reshape([self.num_basis]),
            bias=np.array(self.b.value).squeeze(),
        )
