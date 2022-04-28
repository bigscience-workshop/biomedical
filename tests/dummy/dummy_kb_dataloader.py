#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements a dummy KB dataloader to verify implementation of tests
"""

import datasets

from tests.dummy import dummy_features
from utils.configs import BigBioConfig
from utils.constants import Tasks
from utils.schemas import kb_features

_CITATION = ""
_DATASETNAME = ""
_DESCRIPTION = ""
_HOMEPAGE = ""
_LICENSE = ""

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
    Tasks.RELATION_EXTRACTION,
    Tasks.COREFERENCE_RESOLUTION,
    Tasks.EVENT_EXTRACTION,
]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


_DATASETS_SPLITS = {
    "train": datasets.Split.TRAIN,
    "dev": datasets.Split.VALIDATION,
    "test": datasets.Split.TEST,
}

_GLOBAL_UNIQUE_IDS = {
    "splits": ["train", "dev", "test"],
    "examples": [
        {
            "id": 0,
            "document_id": 0,
            "passages": [dummy_features.DummyPassage(uid=0)],
            "entities": [dummy_features.DummyEntity(uid=1)],
            "relations": [],
            "coreferences": [],
            "events": [],
        }
    ],
}

TESTS = {"global_unique_ids": _GLOBAL_UNIQUE_IDS}


class DummyDataset(datasets.GeneratorBasedBuilder):
    """
    BioCreative V Chemical Disease Relation (CDR) Task.
    """

    DEFAULT_CONFIG_NAME = "global_unique_ids"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=name + "_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=name,
            schema="source",
            subset_id=name,
        )
        for name, specs in TESTS.items()
    ]

    BUILDER_CONFIGS += [
        BigBioConfig(
            name=name + "_bigbio_kb",
            version=datasets.Version(_SOURCE_VERSION),
            description=name,
            schema="bigbio_kb",
            subset_id=name,
        )
        for name, specs in TESTS.items()
    ]

    def _info(self):

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=kb_features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    @property
    def test_name(self):

        return self.config.name.replace("_source", "").replace("_bigbio_kb", "")

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        specs = TESTS[self.test_name]

        return [
            datasets.SplitGenerator(
                name=_DATASETS_SPLITS[split],
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": split,
                },
            )
            for split in specs["splits"]
        ]

    def _generate_examples(
        self,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""

        specs = TESTS[self.test_name]

        examples = specs["examples"]

        for example in examples:

            yield example["id"], example
