"""
Utility for filtering and loading BigBio datasets.
"""

from importlib.machinery import SourceFileLoader
import logging
import os
import pathlib
from types import ModuleType
from typing import Callable, Iterable, List, Optional

from dataclasses import dataclass
from dataclasses import field
import datasets
from datasets import load_dataset

from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks, SCHEMA_TO_TASKS


# TODO: update this as fixes come in
_CURRENTLY_BROKEN_NAMES = set(
    [
        "nagel_source",  # download url
        "nagel_bigbio_kb",  # download url
        "pcr_source",  # download url
        "pcr_fixed_source",  # download url
        "pcr_bigbio_kb",  # download url
    ]
)


# large datasets take greater than ~ 10 minutes to load
_LARGE_CONFIG_NAMES = set(
    [
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
    ]
)


# resource datasets are widely used but not annotated by human experts
# e.g. PubTator and MIMIC III
_RESOURCE_CONFIG_NAMES = set(
    [
        "pubtator_central_sample_source",
        "pubtator_central_sample_bigbio_kb",
        "pubtator_central_source",
        "pubtator_central_bigbio_kb",
    ]
)


# BLURB benchmark datasets
# https://microsoft.github.io/BLURB/tasks.html
# ==========================================================
_BLURB_CONFIG_NAMES = set(
    "bc5cdr_bigbio_kb",  # BC5-chem and BC5-disease
    "biosses_bigbio_pairs",  # BIOSSES
    "ebm_pico_bigbio_kb",  # EBM PICO
    "gnormplus_bigbio_kb",  # BC2GM
    "jnlpba_bigbio_kb",  # JNLPBA
    "ncbi_disease_bigbio_kb",  # NCBI-disease
    "bioasq_7b_bigbio_qa",  # BioASQ
    "chemprot_bigbio_kb",  # ChemProt
    "ddi_corpus_bigbio_kb",  # DDI
    "hallmarks_of_cancer_bigbio_text",  # HOC
    #    ''                                   #  Gene-Disease Associations (GAD)
    "pubmed_qa_labeled_fold0_bigbio_qa",  # PubMedQA
)


@dataclass
class BigBioConfigHelper:
    """Metadata for one config of a dataset."""

    script: pathlib.Path
    dataset_name: str
    tasks: List[Tasks]
    config: BigBioConfig
    is_local: bool
    is_bigbio_schema: bool
    bigbio_schema_caps: Optional[str]
    is_large: bool
    is_resource: bool
    is_default: bool
    is_broken: bool
    bigbio_version: str
    source_version: str
    citation: str
    description: str
    homepage: str
    license: str

    _ds_module: datasets.load.DatasetModule = field(repr=False)
    _py_module: ModuleType = field(repr=False)
    _ds_cls: type = field(repr=False)

    def load_dataset(
        self,
        **load_dataset_kwargs,
    ):
        return load_dataset(
            path=self.script,
            name=self.config.name,
            **load_dataset_kwargs,
        )


def default_is_keeper(helper: BigBioConfigHelper) -> bool:
    return not helper.is_large and not helper.is_resource and helper.is_bigbio_schema


class BigBioConfigHelpers:
    """
    Handles creating and filtering BigBioDatasetConfigHelper instances.
    """

    def __init__(
        self,
        helpers: Optional[Iterable[BigBioConfigHelper]] = None,
        keep_broken: bool = False,
    ):

        path_to_here = pathlib.Path(__file__).parent.absolute()
        self.path_to_biodatasets = (path_to_here / "biodatasets").resolve()
        self.dataloader_scripts = sorted(
            self.path_to_biodatasets.glob(os.path.join("*", "*.py"))
        )
        self.dataloader_scripts = [
            el for el in self.dataloader_scripts if el.name != "__init__.py"
        ]

        # if helpers are passed in, just attach and go
        if helpers is not None:
            if keep_broken:
                self._helpers = helpers
            else:
                self._helpers = [helper for helper in helpers if not helper.is_broken]
            return

        # otherwise, create all helpers available in package
        helpers = []
        for dataloader_script in self.dataloader_scripts:
            dataset_name = dataloader_script.stem
            py_module = SourceFileLoader(
                dataset_name, dataloader_script.as_posix()
            ).load_module()
            ds_module = datasets.load.dataset_module_factory(
                dataloader_script.as_posix()
            )
            ds_cls = datasets.load.import_main_class(ds_module.module_path)

            for config in ds_cls.BUILDER_CONFIGS:

                is_bigbio_schema = config.schema.startswith("bigbio")
                if is_bigbio_schema:
                    bigbio_schema_caps = config.schema.split("_")[1].upper()
                    tasks = SCHEMA_TO_TASKS[bigbio_schema_caps] & set(
                        py_module._SUPPORTED_TASKS
                    )
                else:
                    tasks = py_module._SUPPORTED_TASKS
                    bigbio_schema_caps = None

                helpers.append(
                    BigBioConfigHelper(
                        script=dataloader_script.as_posix(),
                        dataset_name=dataset_name,
                        tasks=tasks,
                        config=config,
                        is_local=py_module._LOCAL,
                        is_bigbio_schema=is_bigbio_schema,
                        bigbio_schema_caps=bigbio_schema_caps,
                        is_large=config.name in _LARGE_CONFIG_NAMES,
                        is_resource=config.name in _RESOURCE_CONFIG_NAMES,
                        is_default=config.name == ds_cls.DEFAULT_CONFIG_NAME,
                        is_broken=config.name in _CURRENTLY_BROKEN_NAMES,
                        bigbio_version=py_module._BIGBIO_VERSION,
                        source_version=py_module._SOURCE_VERSION,
                        citation=py_module._CITATION,
                        description=py_module._DESCRIPTION,
                        homepage=py_module._HOMEPAGE,
                        license=py_module._LICENSE,
                        _ds_module=ds_module,
                        _py_module=py_module,
                        _ds_cls=ds_cls,
                    )
                )

        if keep_broken:
            self._helpers = helpers
        else:
            self._helpers = [helper for helper in helpers if not helper.is_broken]

    @property
    def available_dataset_names(self) -> List[str]:
        return sorted(list(set([helper.dataset_name for helper in self])))

    def for_dataset(self, dataset_name: str) -> "BigBioConfigHelpers":
        helpers = [helper for helper in self if helper.dataset_name == dataset_name]
        return BigBioConfigHelpers(helpers=helpers)

    def default_for_dataset(self, dataset_name: str) -> BigBioConfigHelper:
        helpers = [
            helper
            for helper in self
            if helper.is_default and helper.dataset_name == dataset_name
        ]
        assert len(helpers) == 1
        return helpers[0]

    def filtered(
        self, is_keeper: Callable[[BigBioConfigHelper], bool]
    ) -> "BigBioConfigHelpers":
        """Return dataset config helpers that match is_keeper."""
        return BigBioConfigHelpers(
            helpers=[helper for helper in self if is_keeper(helper)]
        )

    def __iter__(self):
        for helper in self._helpers:
            yield helper

    def __len__(self):
        return len(self._helpers)


if __name__ == "__main__":

    conhelps = BigBioConfigHelpers()

    # filter and load datasets
    # ====================================================================
    tmvar_datasets = [
        load_dataset(**helper.get_load_dataset_kwargs())
        for helper in conhelps.filtered(
            lambda x: ("tmvar" in x.dataset_name and x.is_bigbio_schema)
        )
    ]

    # examples of other filters
    # ====================================================================

    # get all source schema config helpers
    source_helpers = conhelps.filtered(lambda x: x.config.schema == "source")

    # get all local bigbio config helpers
    bb_local_helpers = conhelps.filtered(lambda x: x.is_bigbio_schema and x.is_local)

    # bigbio NER public tasks
    bb_ner_public_helpers = conhelps.filtered(
        lambda x: (
            x.is_bigbio_schema
            and Tasks.NAMED_ENTITY_RECOGNITION in x.tasks
            and not x.is_local
        )
    )

    # n2c2 datasets
    bb_n2c2_helpers = conhelps.filtered(
        lambda x: ("n2c2" in x.dataset_name and x.is_bigbio_schema)
    )
