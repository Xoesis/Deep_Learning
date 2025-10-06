import jax
import jax.numpy as jnp
from flax import nnx
import structlog

log = structlog.get_logger()


class Conv2d(nnx.Module):
    def __init__(
        self,
        key,
        l2pen: int,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding="SAME",
    ):
        self.rngs = nnx.Rngs(params=key)
        self.l2pen = l2pen
        self.padding = padding
        self.layer = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            rngs=self.rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.layer(x)
        return x

    def l2_loss(self) -> jax.Array:
        return self.l2pen * jnp.sum(self.layer.kernel.value**2)


class BatchNorm(nnx.Module):
    def __init__(self, num_channels: int, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.gamma = nnx.Param(jnp.ones((num_channels,)))
        self.beta = nnx.Param(jnp.zeros((num_channels,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape (num_samples, height, width, channels)
        _, _, _, c = x.shape

        # Find mean and variance of each channel
        u = jnp.mean(x, axis=(1, 2), keepdims=True)
        var = jnp.var(x, axis=(1, 2), keepdims=True)

        # Normalize
        x_norm = (x - u) / (jnp.sqrt(jnp.square(var) + self.epsilon))

        # Shift and Scale
        gamma = self.gamma.reshape(1, 1, 1, c)
        beta = self.beta.reshape(1, 1, 1, c)
        x_batch_norm = gamma * x_norm + beta
        return x_batch_norm


class ResBlock(nnx.Module):
    def __init__(
        self,
        key,
        l2pen: int,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding="SAME",
        activation=jax.nn.relu,
    ):
        self.rngs = nnx.Rngs(params=key)
        self.l2pen = l2pen
        self.padding = padding
        if in_channels != out_channels:
            id_key, fl_key = jax.random.split(key, 2)
            self.id_layer_x = Conv2d(
                id_key,
                self.l2pen,
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=stride,
            )
            self.con_x = True
        else:
            fl_key = key
            self.con_x = False

        self.layer1 = Conv2d(
            fl_key,
            self.l2pen,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.batch1 = BatchNorm(out_channels)
        self.activation = activation
        self.layer2 = Conv2d(
            fl_key,
            self.l2pen,
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
        )
        self.batch2 = BatchNorm(out_channels)

    def __call__(self, x: jax.Array) -> jax.Array:
        fl = self.layer1(x)
        fl = self.batch1(fl)
        fl = self.activation(fl)
        fl = self.layer2(fl)
        fl = self.batch2(fl)

        if self.con_x:
            x = self.id_layer_x(x)
        return self.activation(x + fl)

    def l2_loss(self):
        l2loss = self.layer1.l2_loss()
        l2loss += self.layer2.l2_loss()

        if self.con_x:
            l2loss += self.id_layer_x.l2_loss()
        return l2loss


class Classifier(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        stride: list[int],
        num_classes: int,
        l2pen: int,
    ):
        keys = rngs.params()
        self.layer_depths = layer_depths
        self.kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes
        self.input_depth = input_depth
        self.l2pen = l2pen
        self.stride = stride

        self.keys = jax.random.split(keys, len(self.layer_depths))
        self.input_features = [self.input_depth] + self.layer_depths[:-1]
        self.first_layer = Conv2d(
            key=self.keys[0],
            l2pen=self.l2pen,
            in_channels=self.input_features[0],
            out_channels=self.layer_depths[0],
            kernel_size=self.kernel_sizes[0],
            stride=self.stride[0],
        )
        self.layers = []
        for i in range(1, len(self.layer_depths)):
            self.layers.append(
                ResBlock(
                    self.keys[i],
                    self.l2pen,
                    self.input_features[i],
                    self.layer_depths[i],
                    self.kernel_sizes[i],
                    self.stride[i],
                )
            )

        self.final_layer = nnx.Linear(
            in_features=self.layer_depths[-1], out_features=self.num_classes, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = jnp.mean(x, axis=(1, 2))
        x = x.reshape((x.shape[0], -1))
        x = self.final_layer(x)
        return x

    def l2_loss(self):
        l2_loss = 0
        for layer in self.layers:
            l2_loss += layer.l2_loss()
        return l2_loss
