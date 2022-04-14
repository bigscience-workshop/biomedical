from dataclasses import dataclass

import datasets


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None

    def __post_init__(self):
        if self.name is None:
            raise ValueError("pleas pass an explicit config name to load_dataset")
