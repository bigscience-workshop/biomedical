##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta dataloader to load "load configurations" for all available datasets according to `_SUPPORTED_TASKS`
"""

import importlib
import os
from collections import defaultdict
from glob import glob
from types import ModuleType
from typing import Optional, Union

import datasets
from bibgio.utils.constants import Tasks

_SCHEMA_TO_TASKS = {
    "bigbio_kb": [
        Tasks.NAMED_ENTITY_RECOGNITION,
        Tasks.NAMED_ENTITY_DISAMBIGUATION,
        Tasks.EVENT_EXTRACTION,
        Tasks.RELATION_EXTRACTION,
        Tasks.COREFERENCE_RESOLUTION,
    ],
    "bigbio_qa": [Tasks.QUESTION_ANSWERING],
    "bigbio_te": [Tasks.TEXTUAL_ENTAILMENT],
    "bigbio_pairs": [Tasks.SEMANTIC_SIMILARITY],
    "bigbio_t2t": [Tasks.PARAPHRASING, Tasks.TRANSLATION, Tasks.SUMMARIZATION],
    "bigbio_text": [Tasks.TEXT_CLASSIFICATION],
}


class BigBioDataloader:
    """
    Meta dataloader biodatasets
    """

    def __init__(
        self,
        tasks: Optional[list[Tasks]] = None,
        data_dir: Optional[Union[str, dict]] = None,
    ):
        self.scripts = sorted(glob(os.path.join("biodatasets", "*", "*.py")))
        self.tasks = tasks

        # for local datasets:
        # str = global `data_dir`, dict = per script `data_dir`
        self.data_dir = data_dir

    def load_module_from_path(self, path: str) -> ModuleType:
        """
        Load dataloader script as python module
        """

        module = path
        if module.endswith(".py"):
            module = module[:-3]
        module = module.replace("/", ".")

        return importlib.import_module(module)

    def get_supported_tasks(self, module: ModuleType) -> list[Tasks]:
        """
        Wrapper to get supported tasks
        """

        return module._SUPPORTED_TASKS

    def get_for_load_dataset(self) -> dict:
        """
        Loads kwargs for `datasets.load_dataset` for all datasets according to tasks
        """

        bigbio_datasets = defaultdict(list)

        for script in self.scripts:

            module = self.load_module_from_path(script)

            supported_tasks = self.get_supported_tasks(module)

            # skip dataset if tasks are specified
            # and dataset does not have at least one selected supported task
            if self.tasks is not None and not any(
                t in self.tasks for t in supported_tasks
            ):
                continue

            datasets_module = datasets.load.dataset_module_factory(script)

            builder_cls = datasets.load.import_main_class(datasets_module.module_path)

            configs = [
                config
                for config in builder_cls.BUILDER_CONFIGS
                if config.schema.startswith("bigbio")
            ]

            # subsets may have mutually exclusive supported tasks, e.g. one subset KB and another TEXT
            for config in configs:

                config_tasks = _SCHEMA_TO_TASKS[config.schema]

                if self.tasks is not None and not any(
                    ct in self.tasks for ct in config_tasks
                ):
                    continue

                bigbio_datasets[config.schema].append(
                    {"path": script, "name": config.name}
                )

        return bigbio_datasets


if __name__ == "__main__":

    md = BigBioDataloader()

    bibgio_datasets = md.get_for_load_dataset()
