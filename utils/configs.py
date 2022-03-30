from typing import Union

import datasets
from dataclasses import dataclass


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    name: str = None
    version: Union[datasets.Version, str] = None
    description: str = None
    schema: str = None
    subset_id: str = None
