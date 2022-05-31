"""
Utility for filtering and loading BigBio datasets.
"""
from collections import Counter
from importlib.machinery import SourceFileLoader
import logging
import os
import pathlib
from types import ModuleType
from typing import Callable, Iterable, List, Optional, Dict

from dataclasses import dataclass
from dataclasses import field
import datasets
from datasets import load_dataset

from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks, SCHEMA_TO_TASKS, Lang


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
    [
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
        "gad_fold0_bigbio_text",  #  Gene-Disease Associations (GAD)
        "pubmed_qa_labeled_fold0_bigbio_qa",  # PubMedQA
    ]
)


MAX_COMMON = 50


@dataclass
class BigBioKbMetadata:

    samples_count: int

    passages_count: int
    passages_char_count: int
    passages_type_counter: Dict[str, int]

    entities_count: int
    entities_type_counter: Dict[str, int]
    entities_db_name_counter: Dict[str, int]
    entities_unique_db_ids_count: int

    events_count: int
    events_type_counter: Dict[str, int]
    events_arguments_count: 0
    events_arguments_role_counter: Dict[str, int]

    coreferences_count: 0

    relations_count: 0
    relations_type_counter: Counter()
    relations_db_name_counter: Counter()
    relations_unique_db_ids_count: int

    @classmethod
    def from_dataset(cls, ds, max_common=MAX_COMMON):

        passages_count = 0
        passages_char_count = 0
        passages_type_counter = Counter()

        entities_count = 0
        entities_type_counter = Counter()
        entities_db_name_counter = Counter()
        entities_unique_db_ids = set()

        events_count = 0
        events_type_counter = Counter()
        events_arguments_count = 0
        events_arguments_role_counter = Counter()

        coreferences_count = 0

        relations_count = 0
        relations_type_counter = Counter()
        relations_db_name_counter = Counter()
        relations_unique_db_ids = set()

        for sample in ds:
            for passage in sample["passages"]:
                passages_count += 1
                passages_char_count += len(passage["text"][0])
                passages_type_counter[passage["type"]] += 1

            for entity in sample["entities"]:
                entities_count += 1
                entities_type_counter[entity["type"]] += 1
                for norm in entity["normalized"]:
                    entities_db_name_counter[norm["db_name"]] += 1
                    entities_unique_db_ids.add(norm["db_id"])

            for event in sample["events"]:
                events_count += 1
                events_type_counter[event["type"]] += 1
                for argument in event["arguments"]:
                    events_arguments_count += 1
                    events_arguments_role_counter[argument["role"]] += 1

            for coreference in sample["coreferences"]:
                coreferences_count += 1

            for relation in sample["relations"]:
                relations_count += 1
                relations_type_counter[relation["type"]] += 1
                for norm in relation["normalized"]:
                    relations_db_name_counter[norm["db_name"]] += 1
                    relations_unique_db_ids.add(norm["db_id"])

        for cc in [
            passages_type_counter,
            entities_type_counter,
            entities_db_name_counter,
            events_arguments_role_counter,
            relations_type_counter,
        ]:
            if None in cc.keys():
                raise ValueError()

        return cls(
            samples_count=ds.num_rows,
            passages_count=passages_count,
            passages_type_counter=dict(passages_type_counter.most_common(max_common)),
            passages_char_count=passages_char_count,
            entities_count=entities_count,
            entities_type_counter=dict(entities_type_counter.most_common(max_common)),
            entities_db_name_counter=dict(
                entities_db_name_counter.most_common(max_common)
            ),
            entities_unique_db_ids_count=len(entities_unique_db_ids),
            events_count=events_count,
            events_type_counter=dict(events_type_counter.most_common(max_common)),
            events_arguments_count=events_arguments_count,
            events_arguments_role_counter=dict(
                events_arguments_role_counter.most_common(max_common)
            ),
            coreferences_count=coreferences_count,
            relations_count=relations_count,
            relations_type_counter=dict(relations_type_counter.most_common(max_common)),
            relations_db_name_counter=dict(
                relations_db_name_counter.most_common(max_common)
            ),
            relations_unique_db_ids_count=len(relations_unique_db_ids),
        )


@dataclass
class BigBioTextMetadata:

    samples_count: int
    text_char_count: int
    labels_count: int
    labels_counter: Dict[str, int]

    @classmethod
    def from_dataset(cls, ds, max_common=MAX_COMMON):

        text_char_count = 0
        labels_count = 0
        labels_counter = Counter()

        for sample in ds:
            text_char_count += len(sample["text"]) if sample["text"] is not None else 0
            for label in sample["labels"]:
                labels_count += 1
                labels_counter[label] += 1

        return cls(
            samples_count=ds.num_rows,
            text_char_count=text_char_count,
            labels_count=labels_count,
            labels_counter=dict(labels_counter.most_common(max_common)),
        )


@dataclass
class BigBioPairsMetadata:

    samples_count: int
    text_1_char_count: int
    text_2_char_count: int
    label_counter: Dict[str, int]

    @classmethod
    def from_dataset(cls, ds, max_common=MAX_COMMON):

        text_1_char_count = 0
        text_2_char_count = 0
        label_counter = Counter()

        for sample in ds:
            text_1_char_count += (
                len(sample["text_1"]) if sample["text_1"] is not None else 0
            )
            text_2_char_count += (
                len(sample["text_2"]) if sample["text_2"] is not None else 0
            )
            label_counter[sample["label"]] += 1

        return cls(
            samples_count=ds.num_rows,
            text_1_char_count=text_1_char_count,
            text_2_char_count=text_2_char_count,
            label_counter=dict(label_counter.most_common(max_common)),
        )


@dataclass
class BigBioQaMetadata:

    samples_count: int
    question_char_count: int
    context_char_count: int
    answer_count: int
    answer_char_count: int
    type_counter: Dict[str, int]
    choices_counter: Dict[str, int]

    @classmethod
    def from_dataset(cls, ds, max_common=MAX_COMMON):

        question_char_count = 0
        context_char_count = 0
        answer_count = 0
        answer_char_count = 0
        type_counter = Counter()
        choices_counter = Counter()

        for sample in ds:
            question_char_count += len(sample["question"])
            context_char_count += len(sample["context"])
            type_counter[sample["type"]] += 1
            for choice in sample["choices"]:
                choices_counter[choice] += 1
            for answer in sample["answer"]:
                answer_count += 1
                answer_char_count += len(answer)

        return cls(
            samples_count=ds.num_rows,
            question_char_count=question_char_count,
            context_char_count=context_char_count,
            answer_count=answer_count,
            answer_char_count=answer_char_count,
            type_counter=dict(type_counter.most_common(max_common)),
            choices_counter=dict(choices_counter.most_common(max_common)),
        )


@dataclass
class BigBioT2tMetadata:

    samples_count: int
    text_1_char_count: int
    text_2_char_count: int
    text_1_name_counter: Dict[str, int]
    text_2_name_counter: Dict[str, int]

    @classmethod
    def from_dataset(cls, ds, max_common=MAX_COMMON):

        text_1_char_count = 0
        text_2_char_count = 0
        text_1_name_counter = Counter()
        text_2_name_counter = Counter()

        for sample in ds:
            text_1_char_count += (
                len(sample["text_1"]) if sample["text_1"] is not None else 0
            )
            text_2_char_count += (
                len(sample["text_2"]) if sample["text_2"] is not None else 0
            )
            text_1_name_counter[sample["text_1_name"]] += 1
            text_2_name_counter[sample["text_2_name"]] += 1

        return cls(
            samples_count=ds.num_rows,
            text_1_char_count=text_1_char_count,
            text_2_char_count=text_2_char_count,
            text_1_name_counter=dict(text_1_name_counter.most_common(max_common)),
            text_2_name_counter=dict(text_2_name_counter.most_common(max_common)),
        )


@dataclass
class BigBioTeMetadata:

    samples_count: int
    premise_char_count: int
    hypothesis_char_count: int
    label_counter: Dict[str, int]

    @classmethod
    def from_dataset(cls, ds, max_common=MAX_COMMON):

        premise_char_count = 0
        hypothesis_char_count = 0
        label_counter = Counter()

        for sample in ds:
            premise_char_count += (
                len(sample["premise"]) if sample["premise"] is not None else 0
            )
            hypothesis_char_count += (
                len(sample["hypothesis"]) if sample["hypothesis"] is not None else 0
            )
            label_counter[sample["label"]] += 1

        return cls(
            samples_count=ds.num_rows,
            premise_char_count=premise_char_count,
            hypothesis_char_count=hypothesis_char_count,
            label_counter=dict(label_counter.most_common(max_common)),
        )


SCHEMA_TO_METADATA_CLS = {
    "bigbio_kb": BigBioKbMetadata,
    "bigbio_text": BigBioTextMetadata,
    "bigbio_pairs": BigBioPairsMetadata,
    "bigbio_qa": BigBioQaMetadata,
    "bigbio_t2t": BigBioT2tMetadata,
    "bigbio_te": BigBioTeMetadata,
}


@dataclass
class BigBioConfigHelper:
    """Metadata for one config of a dataset."""

    script: pathlib.Path
    dataset_name: str
    tasks: List[Tasks]
    languages: List[Lang]
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

    def get_load_dataset_kwargs(
        self,
        **extra_load_dataset_kwargs,
    ):
        return {
            "path": self.script,
            "name": self.config.name,
            **extra_load_dataset_kwargs,
        }

    def load_dataset(
        self,
        **extra_load_dataset_kwargs,
    ):
        return load_dataset(
            path=self.script,
            name=self.config.name,
            **extra_load_dataset_kwargs,
        )

    def get_metadata(self, **extra_load_dataset_kwargs):
        if not self.is_bigbio_schema:
            raise ValueError("only supported for bigbio schemas")
        dsd = self.load_dataset(**extra_load_dataset_kwargs)
        split_metas = {}
        for split, ds in dsd.items():
            meta = SCHEMA_TO_METADATA_CLS[self.config.schema].from_dataset(ds)
            split_metas[split] = meta
        return split_metas


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
                        languages=py_module._LANGUAGES,
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
        if len(helpers) == 0:
            raise ValueError(f"no helper with helper.dataset_name = {dataset_name}")
        return BigBioConfigHelpers(helpers=helpers)

    def for_config_name(self, config_name: str) -> "BigBioConfigHelper":
        helpers = [helper for helper in self if helper.config.name == config_name]
        if len(helpers) == 0:
            raise ValueError(f"no helper with helper.config.name = {config_name}")
        if len(helpers) > 1:
            raise ValueError(
                f"multiple helpers with helper.config.name = {config_name}"
            )
        return helpers[0]

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

    def __repr__(self):
        return "\n\n".join([helper.__repr__() for helper in self])

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for helper in self._helpers:
            yield helper

    def __len__(self):
        return len(self._helpers)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return BigBioConfigHelpers(
                helpers=[self._helpers[ii] for ii in range(start, stop, step)]
            )
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index ({key}) is out of range.")
            return self._helpers[key]
        else:
            raise TypeError("Invalid argument type.")


if __name__ == "__main__":

    conhelps = BigBioConfigHelpers()

    # filter and load datasets
    # ====================================================================
    tmvar_datasets = [
        helper.load_dataset()
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
