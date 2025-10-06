from importlib.resources import files
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataSettings(BaseModel):
    """Settings for data generation."""

    cifar10or100: bool = True
    train_val_split: float = 0.9


class ModelSettings(BaseModel):
    """Settings for the model."""

    l2pen: float = 0.001
    num_classes: int = 10
    input_depth: int = 3
    data_shape: list[int] = [32, 32]
    layer_depth: list[int] = [16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 128, 128]
    kernel: list[Tuple[int, int]] = [
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]
    stride: list[int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


class TrainingSettings(BaseModel):
    """Settings for model training."""

    batch_size: int = 128
    num_iters: int = 50000
    learning_rate: float = 0.001


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw04").joinpath("config.toml"),
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Set the priority of settings sources.

        We use a TOML file for configuration.
        """
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
