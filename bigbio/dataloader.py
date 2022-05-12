"""
Utility for filtering and loading BigBio datasets.
"""

from importlib.machinery import SourceFileLoader
import logging
import os
import pathlib
from types import ModuleType
from typing import Callable, List, Optional

from dataclasses import dataclass
import datasets
from datasets import load_dataset

from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks, SCHEMA_TO_TASKS


# TODO: maybe move this to bigbio.utils.constants
# large datasets take greater than approximately 5 minutes to load
_LARGE_CONFIG_NAMES = set([
    "biomrc_large_A_source",
    "biomrc_large_B_source",
    "biomrc_large_A_bigbio_qa",
    "biomrc_large_B_bigbio_qa",

    "medal_source",
    "medal_bigbio_kb",

    "meddialog_zh_source",
    "meddialog_zh_bigbio_text",

    "pubtator_central_source",
    "pubtator_central_bigbio_kb",
])


# TODO: maybe move this to bigbio.utils.constants
# resource datasets are widely used but not expertly annotated
# e.g. PubTator and MIMIC III
_RESOURCE_CONFIG_NAMES = set([
    "pubtator_central_sample_source",
    "pubtator_central_sample_bigbio_kb",
    "pubtator_central_source",
    "pubtator_central_bigbio_kb",
])


@dataclass
class DatasetConfigHelper:
    """Metadata for one config of a dataset."""
    script: pathlib.Path
    dataset_name: str
    py_module: ModuleType
    ds_module: datasets.load.DatasetModule
    ds_cls: type
    tasks: List[Tasks]
    config: BigBioConfig
    is_local: bool
    is_bigbio_schema: bool
    bigbio_schema_caps: Optional[str]
    is_large: bool
    is_resource: bool


class BigBioDataloader:
    """
    Meta dataloader biodatasets
    """

    def __init__(self):

        path_to_here = pathlib.Path(__file__).parent.absolute()
        self.path_to_biodatasets = (path_to_here / ".." / "biodatasets").resolve()
        self.dataloader_scripts = sorted(self.path_to_biodatasets.glob(os.path.join("*", "*.py")))

        ds_config_helpers = []
        for dataloader_script in self.dataloader_scripts:
            dataset_name = dataloader_script.stem
            py_module = SourceFileLoader(dataset_name, dataloader_script.as_posix()).load_module()
            ds_module = datasets.load.dataset_module_factory(dataloader_script.as_posix())
            ds_cls = datasets.load.import_main_class(ds_module.module_path)

            for config in ds_cls.BUILDER_CONFIGS:
                is_bigbio_schema = config.schema.startswith("bigbio")
                if is_bigbio_schema:
                    bigbio_schema_caps = config.schema.split("_")[1].upper()
                    tasks = SCHEMA_TO_TASKS[bigbio_schema_caps] & set(py_module._SUPPORTED_TASKS)
                else:
                    tasks = py_module._SUPPORTED_TASKS
                    bigbio_schema_caps = None

                ds_config_helpers.append(
                    DatasetConfigHelper(
                        script=dataloader_script.as_posix(),
                        dataset_name=dataset_name,
                        py_module=py_module,
                        ds_module=ds_module,
                        ds_cls=ds_cls,
                        tasks=tasks,
                        config=config,
                        is_local=py_module._LOCAL,
                        is_bigbio_schema=is_bigbio_schema,
                        bigbio_schema_caps=bigbio_schema_caps,
                        is_large=config.name in _LARGE_CONFIG_NAMES,
                        is_resource=config.name in _RESOURCE_CONFIG_NAMES,
                    )
                )

        self.ds_config_helpers = ds_config_helpers


    def get_filtered_config_helpers(
        self,
        is_keeper: Callable[[DatasetConfigHelper], bool]
    ) -> DatasetConfigHelper:
        """Return dataset config helpers that match is_keeper."""
        return [
            dch for dch in self.ds_config_helpers
            if is_keeper(dch)
        ]


    @staticmethod
    def get_load_dataset_kwargs_from_config_helper(
        ds_config_helper,
        local_data_dir: Optional[str] = None,
    ):

        return {
            'path': ds_config_helper.script,
            'name': ds_config_helper.config.name,
            'data_dir': local_data_dir,
        }



if __name__ == "__main__":

    bigbio_dataloader = BigBioDataloader()


    # first, define is_keeper function and filter config helpers
    # second, get load_datset kwargs
    # third, load dataset
    #====================================================================
    bb_tmvar_helpers = bigbio_dataloader.get_filtered_config_helpers(
        lambda x: (
            "tmvar" in x.dataset_name and
            x.is_bigbio_schema
        )
    )
    load_ds_kwargs = [
        bigbio_dataloader.get_load_dataset_kwargs_from_config_helper(helper)
        for helper in bb_tmvar_helpers
    ]
    tmvar_datasets = [load_dataset(**kwargs) for kwargs in load_ds_kwargs]


    # examples of other filters
    #====================================================================

    # get all config helpers
    all_helpers = bigbio_dataloader.get_filtered_config_helpers(
        lambda x: True)

    # get all source schema config helpers
    source_helpers = bigbio_dataloader.get_filtered_config_helpers(
        lambda x: x.config.schema == "source")

    # get all local bigbio config helpers
    bb_local_helpers = bigbio_dataloader.get_filtered_config_helpers(
        lambda x: x.is_bigbio_schema and x.is_local
    )

    # bigbio NER public tasks
    bb_ner_public_helpers = bigbio_dataloader.get_filtered_config_helpers(
        lambda x: (
            x.is_bigbio_schema and
            Tasks.NAMED_ENTITY_RECOGNITION in x.tasks and
            not x.is_local
        )
    )

    # n2c2 datasets
    bb_n2c2_helpers = bigbio_dataloader.get_filtered_config_helpers(
        lambda x: (
            "n2c2" in x.dataset_name and
            x.is_bigbio_schema
        )
    )
