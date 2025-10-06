from dataclasses import InitVar, dataclass, field

import numpy as np
import tensorflow as tf


@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    train_val_split: float = 0.9
    cifar10or100: bool = True
    x_train: np.ndarray = field(init=False)
    y_train: np.ndarray = field(init=False)
    x_val: np.ndarray = field(init=False)
    y_val: np.ndarray = field(init=False)
    x_test: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)
    """Like Ceaser split all gaul into three parts we split the data into 3 parts train, val, and test"""

    def __post_init__(
        self,
        rng: np.random.Generator,
        train_val_split: float = 0.9,
        cifar10or100: bool = True,
    ):
        """Generate synthetic data based on the model."""
        if cifar10or100:
            (x_train_temp, y_train_temp), (self.x_test, self.y_test) = (
                tf.keras.datasets.cifar10.load_data()
            )
        else:
            (x_train_temp, y_train_temp), (self.x_test, self.y_test) = (
                tf.keras.datasets.cifar100.load_data()
            )
        """ x_train_temp = (50000, 32, 32, 3),
            y_train_temp = (50000, 1),
            self.x_test = (10000, 32, 32, 3),
            self.y_test = (10000, 1)"""

        x_train_temp = x_train_temp / 255.0  # (50000, 32, 32, 3)
        self.x_test = self.x_test / 255.0

        """Split the data into train and val sets randomly"""
        self.train_val_split = train_val_split
        indices = np.arange(len(y_train_temp))
        rng.shuffle(indices)
        split_idx = int(len(y_train_temp) * self.train_val_split)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        self.x_train = x_train_temp[train_idx]
        self.y_train = y_train_temp[train_idx]
        self.x_val = x_train_temp[val_idx]
        self.y_val = y_train_temp[val_idx]
        self.index = np.arange(len(self.y_train))

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x_train[choices], self.y_train[choices]

    def get_val(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the entire validation set."""
        return self.x_val, self.y_val

    def get_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the entire test set."""
        return self.x_test, self.y_test
