from dataclasses import dataclass, asdict
from functools import partial
from typing import List, Callable
import datasets


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None


def default_BigBioConfig(
    DEFAULT_CONFIG_NAME: str,
    BUILDER_CONFIGS: List[BigBioConfig],
    ) -> Callable:
    """
    If config_args specified (i.e. local datasets), create a partial class for `BUILDER_CONFIG_CLASS` that will inherit the default view.
    
    Args:
        DEFAULT_CONFIG_NAME: Name of the default config
        BUILDER_CONFIGS: Default configs available in the dataclass
    """  # noqa
    for config in BUILDER_CONFIGS:
        if config.name == DEFAULT_CONFIG_NAME:
            config_args = asdict(config)

    return partial(BigBioConfig, **config_args)
