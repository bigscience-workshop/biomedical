##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta dataloader to load "load configurations" for all available datasets according to `_SUPPORTED_TASKS`
"""

import importlib
import os
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from types import ModuleType
from typing import Optional, Union

import datasets
from loguru import logger

from bigbio.utils.constants import Tasks

_SCHEMA_TO_TASKS: dict = {
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


# NOTE:
# huge datasets are those for which loading takes >5 minutes
_HUGE: dict = {
    "biomrc": [
        "biomrc_large_B_bigbio_qa",
        "biomrc_large_A_bigbio_qa",
    ],
    "medal": ["medal_bigbio_kb"],
    "meddialog": ["meddialog_zh_bigbio_text"],
    "pubtator_central": [
        "pubtator_central_bigbio_kb",
    ],
}


# NOTE:
# weak datasets are the unstructured ones,
# i.e. not providing a clear task but being more of a resource,
# e.g. PubTator and MIMICIII
_WEAK: dict = {
    "pubtator_central": [
        "pubtator_central_sample_bigbio_kb",
        "pubtator_central_bigbio_kb",
    ]
}


class BigBioDataloader:
    """
    Meta dataloader biodatasets
    """

    def __init__(
        self,
        tasks: Optional[list[Tasks]] = None,
        data_dir: Optional[str] = None,
        skip_huge: bool = True,
        skip_weak: bool = True,
        skip_local: bool = True,
        exclude: Optional[list] = None,
    ):
        self.scripts = sorted(glob(os.path.join("biodatasets", "*", "*.py")))
        self.tasks = tasks

        # NOTE:
        # my suggestion would be to force all LOCAL datasets
        # to hard-code the upper folder name (where all files are contained)
        # this way we can define a single `data_dir`
        self.data_dir = data_dir

        self.skip_huge = skip_huge
        self.skip_weak = skip_weak
        self.skip_local = skip_local
        self.exclude = (
            [Reference(s).to_name() for s in exclude] if exclude is not None else []
        )

    # pylint: disable=no-self-use
    def load_module(self, ref: str) -> ModuleType:
        """
        Load dataloader script as python module
        """

        return importlib.import_module(Reference(ref).to_module())

    def get_supported_tasks(self, module: Union[ModuleType, str]) -> list[Tasks]:
        """
        Wrapper to get _SUPPORTED_TASKS
        """

        if isinstance(module, str):
            module = self.load_module(module)

        return getattr(module, "_SUPPORTED_TASKS", [])

    def _is_local(self, module: Union[ModuleType, str]) -> bool:
        """
        Wrapper to get _LOCAL
        """

        if isinstance(module, str):
            module = self.load_module(module)

        return getattr(module, "_LOCAL", False)

    def _no_valid_tasks(self, tasks: list[Tasks]) -> bool:

        if len(tasks) == 0:
            return True

        # skip dataset if tasks are specified
        # and dataset does not have at least one selected supported task
        return self.tasks is not None and not any(t in self.tasks for t in tasks)

    def _skip_huge(self, script: str, config: str) -> bool:

        name = Reference(script).to_name()

        skip = bool(self.skip_huge and config in _HUGE.get(name, []))

        if not self.skip_huge and skip:
            logger.warning(
                "The dataloader `{}` generates a HUGE dataset. Downloading and processing will take >5 minutes...",
                name,
            )

        return skip

    def _skip_weak(self, script: str, config: str) -> bool:

        name = Reference(script).to_name()

        skip = bool(self.skip_weak and config in _WEAK.get(name, []))

        if not self.skip_huge and skip:
            logger.warning(
                "The dataloader `{}` generates a WEAK dataset. This is not recommended for training...",
                name,
            )

        return skip

    def get_by_task(self) -> dict:
        """
        Loads kwargs for `datasets.load_dataset` for all datasets according to tasks
        """

        bigbio_datasets = defaultdict(list)

        for script in self.scripts:

            if Reference(script).to_name() in self.exclude:
                continue

            module = self.load_module(script)

            is_local = self._is_local(module)

            if is_local:

                if self.skip_local:
                    continue

                if self.data_dir is None:
                    logger.warning(
                        "Cannot load local dataloader `{}` without if `data_dir` is None. Skip...",
                        Reference(module.__name__).to_name(),
                    )
                    continue

            supported_tasks = self.get_supported_tasks(module)

            if self._no_valid_tasks(tasks=supported_tasks):
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

                if self._skip_huge(script=script, config=config.name):
                    continue

                if self._skip_weak(script=script, config=config.name):
                    continue

                config_tasks = _SCHEMA_TO_TASKS[config.schema]

                if self._no_valid_tasks(tasks=config_tasks):
                    continue

                load_kwargs = {"path": script, "name": config.name}

                if is_local:
                    load_kwargs["data_dir"] = self.data_dir

                bigbio_datasets[config.schema].append(load_kwargs)

        return bigbio_datasets


@dataclass
class Reference:
    """
    String reference to a dataloader, handling: paths, module names or datasets names.
    """

    string: str

    def _is_path(self):

        return os.path.exists(self.string)

    def _is_module(self):

        return not self._is_path() and self.string.find(".") > -1

    def _is_name(self):

        return not self._is_path() and not self._is_module()

    def to_module(self):
        """
        Get dataloader module name
        """

        module = self.string

        if self._is_path():

            if module.endswith(".py"):
                module = module[:-3]
            module = module.replace("/", ".")

        elif self._is_name():
            module = ".".join(["biodatasets", self.string, self.string])

        return module

    def to_path(self):
        """
        Get dataloader path
        """

        path = self.string

        if self._is_module():
            path = path.replace(".", "/") + ".py"

        elif self._is_name():
            path = os.path.join("biodatasets", self.string, self.string) + ".py"

        return path

    def to_name(self):
        """
        Get dataloader name
        """

        name = self.string

        if self._is_path():
            name = os.path.basename(self.string).replace(".py", "")

        elif self._is_module():
            name = self.string.split(".")[-1]

        return name


def test_dataloader_name():
    """
    Test string reference conversion
    """

    path = "biodatasets/paramed/paramed.py"
    module = "biodatasets.paramed.paramed"
    name = "paramed"

    dn = Reference(path)
    print("PATH -> MODULE:", module, "==", dn.to_module(), module == dn.to_module())
    print("PATH -> NAME:", name, "==", dn.to_name(), name == dn.to_name())

    dn = Reference(module)
    print("MODULE -> PATH:", path, "==", dn.to_path(), path == dn.to_path())
    print("MODULE -> NAME:", name, "==", dn.to_name(), name == dn.to_name())

    dn = Reference(name)
    print("NAME -> PATH:", path, "==", dn.to_path(), path == dn.to_path())
    print("NAME -> MODULE:", module, "==", dn.to_module(), module == dn.to_module())


if __name__ == "__main__":
    # test_dataloader_name()

    bigbio_dataloader = BigBioDataloader(exclude=["bc7_litcovid"])

    dataloaders = bigbio_dataloader.get_by_task()

    breakpoint()
