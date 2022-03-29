# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from pathlib import Path

import datasets
import os  # useful for paths
from typing import Iterable, Dict, List
from dataclasses import dataclass
import logging

_DATASETNAME = "bionlp_st_2013_ge"
_UNIFIED_VIEW_NAME = "bigscience"

_CITATION = """\
@inproceedings{kim-etal-2013-genia,
    title = "The {G}enia Event Extraction Shared Task, 2013 Edition - Overview",
    author = "Kim, Jin-Dong  and
      Wang, Yue  and
      Yasunori, Yamamoto",
    booktitle = "Proceedings of the {B}io{NLP} Shared Task 2013 Workshop",
    month = aug,
    year = "2013",
    address = "Sofia, Bulgaria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W13-2002",
    pages = "8--15",
}
"""

_DESCRIPTION = """\
The BioNLP-ST GE task has been promoting development of fine-grained information extraction (IE) from biomedical 
documents, since 2009. Particularly, it has focused on the domain of NFkB as a model domain of Biomedical IE
"""

_HOMEPAGE = "https://github.com/openbiocorpora/bionlp-st-2013-ge"

_LICENSE = "DUA"

_URLs = {"bionlp_st_2013_ge": "https://github.com/openbiocorpora/bionlp-st-2013-ge/archive/refs/heads/master.zip",}

_SUPPORTED_TASKS = ["EE"]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""
    name: str = None
    version: str = None
    description: str = None
    schema: str = None
    subset_id: str = None


class bionlp_st_2013_ge(datasets.GeneratorBasedBuilder):
    """The BioNLP-ST GE task has been promoting development of fine-grained information extraction (IE) from biomedical
    documents, since 2009. Particularly, it has focused on the domain of NFkB as a model domain of Biomedical IE"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bionlp_st_2013_ge_source",
            version=SOURCE_VERSION,
            description=_DESCRIPTION,
            schema='source',
        ),
        BigBioConfig(
            name="bionlp_st_2013_ge_bigbio_kb",
            version=BIGBIO_VERSION,
            description=_DESCRIPTION,
            schema="bigbio_kb",
        ),
    ]


    DEFAULT_CONFIG_NAME = _DATASETNAME

    _ENTITY_TYPES = {
        "Anatomical_system",
        "Cell",
        "Cellular_component",
        "DNA_domain_or_region",
        "Developing_anatomical_structure",
        "Drug_or_compound",
        "Gene_or_gene_product",
        "Immaterial_anatomical_entity",
        "Multi-tissue_structure",
        "Organ",
        "Organism",
        "Organism_subdivision",
        "Organism_substance",
        "Pathological_formation",
        "Protein_domain_or_region",
        "Tissue",
    }

    def _info(self):
        """
        - `features` defines the schema of the parsed data set. The schema depends on the
        chosen `config`: If it is `_SOURCE_VIEW_NAME` the schema is the schema of the
        original data. If `config` is `_UNIFIED_VIEW_NAME`, then the schema is the
        canonical KB-task schema defined in `biomedical/schemas/kb.py`.
        """
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "events": [  # E line in brat
                        {
                            "trigger": datasets.Value(
                                "string"
                            ),  # refers to the text_bound_annotation of the trigger,
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arguments": datasets.Sequence(
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ),
                        }
                    ],
                    "relations": [  # R line in brat
                        {
                            "id": datasets.Value("string"),
                            "head": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "tail": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "type": datasets.Value("string"),
                        }
                    ],
                    "equivalences": [  # Equiv line in brat
                        {
                            "id": datasets.Value("string"),
                            "ref_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                    "attributes": [  # M or A lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                    "normalizations": [  # N lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value(
                                "string"
                            ),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value(
                                "string"
                            ),  # ID in the resource, e.g. 534366
                            "text": datasets.Value(
                                "string"
                            ),  # Human readable description/name of the entity, e.g. "Barack Obama"
                        }
                    ],
                },
            )
        elif self.config.schema == "bigbio_kb":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "passages": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "events": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            # refers to the text_bound_annotation of the trigger
                            "trigger": {
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                                "text": datasets.Sequence(datasets.Value("string"))
                            },
                            "arguments": [
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "coreferences": [
                        {
                            "id": datasets.Value("string"),
                            "entity_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # This is not applicable for MLEE.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        my_urls = _URLs[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": data_dir / f'bionlp-st-2013-ge-master' / "original-data" / "train",
            "dev": data_dir / f'bionlp-st-2013-ge-master' / "original-data" / "devel",
            "test": data_dir / f'bionlp-st-2013-ge-master' / "original-data" / "test",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_files": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_files": data_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_files": data_files["test"]},
            ),
        ]

    def _generate_examples(self, data_files: Path):
        if self.config.schema == "source":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = self.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = self.brat_parse_to_bigbio_kb(
                    self.parse_brat_file(txt_file),
                    entity_types=self._ENTITY_TYPES
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")


    def brat_parse_to_unified_schema(self, brat_parse: Dict) -> Dict:
        unified_example = {}

        # identical
        unified_example["article_id"] = brat_parse["article_id"]
        unified_example["text"] = brat_parse["text"]
        unified_example["events"] = brat_parse["events"]

        # get normalizations
        ref_id_to_normalizations = defaultdict(list)
        for normalization in brat_parse["normalizations"]:
            ref_id_to_normalizations[normalization["ref_id"]].append(
                {
                    "resource_name": normalization["resource_name"],
                    "cuid": normalization["cuid"],
                }
            )

        # separate entities and event triggers
        unified_example["entities"] = []
        unified_example["event_triggers"] = []
        for ann in brat_parse["text_bound_annotations"]:
            if ann["type"] in self._ENTITY_TYPES:
                entity_ann = ann.copy()
                entity_ann["db_refs"] = ref_id_to_normalizations[ann["id"]]
                unified_example["entities"].append(entity_ann)
            else:
                unified_example["event_triggers"].append(ann.copy())

        # massage relations
        unified_example["relations"] = []
        for ann in brat_parse["relations"]:
            unified_example["relations"].append(
                {
                    "head": ann["head"]["ref_id"],
                    "tail": ann["tail"]["ref_id"],
                }
            )

        return unified_example

    def remove_prefix(self, a: str, prefix: str) -> str:
        if a.startswith(prefix):
            a = a[len(prefix):]
        return a

    def parse_brat_file(self, txt_file: Path) -> Dict:
        """
        Parse a brat file into the schema defined below.
        `txt_file` should be the path to the brat '.txt' file you want to parse, e.g. 'data/1234.txt'
        Assumes that the annotations are contained in one or more of the corresponding '.a1', '.a2' or '.ann' files,
        e.g. 'data/1234.ann' or 'data/1234.a1' and 'data/1234.a2'.
        Schema of the parse:
           features = datasets.Features(
                    {
                        "id": datasets.Value("string"),
                        "document_id": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
                            {
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                                "text": datasets.Sequence(datasets.Value("string")),
                                "type": datasets.Value("string"),
                                "id": datasets.Value("string"),
                            }
                        ],
                        "events": [  # E line in brat
                            {
                                "trigger": datasets.Value(
                                    "string"
                                ),  # refers to the text_bound_annotation of the trigger,
                                "id": datasets.Value("string"),
                                "type": datasets.Value("string"),
                                "arguments": datasets.Sequence(
                                    {
                                        "role": datasets.Value("string"),
                                        "ref_id": datasets.Value("string"),
                                    }
                                ),
                            }
                        ],
                        "relations": [  # R line in brat
                            {
                                "id": datasets.Value("string"),
                                "head": {
                                    "ref_id": datasets.Value("string"),
                                    "role": datasets.Value("string"),
                                },
                                "tail": {
                                    "ref_id": datasets.Value("string"),
                                    "role": datasets.Value("string"),
                                },
                                "type": datasets.Value("string"),
                            }
                        ],
                        "equivalences": [  # Equiv line in brat
                            {
                                "id": datasets.Value("string"),
                                "ref_ids": datasets.Sequence(datasets.Value("string")),
                            }
                        ],
                        "attributes": [  # M or A lines in brat
                            {
                                "id": datasets.Value("string"),
                                "type": datasets.Value("string"),
                                "ref_id": datasets.Value("string"),
                                "value": datasets.Value("string"),
                            }
                        ],
                        "normalizations": [  # N lines in brat
                            {
                                "id": datasets.Value("string"),
                                "type": datasets.Value("string"),
                                "ref_id": datasets.Value("string"),
                                "resource_name": datasets.Value(
                                    "string"
                                ),  # Name of the resource, e.g. "Wikipedia"
                                "cuid": datasets.Value(
                                    "string"
                                ),  # ID in the resource, e.g. 534366
                                "text": datasets.Value(
                                    "string"
                                ),  # Human readable description/name of the entity, e.g. "Barack Obama"
                            }
                        ],
                    },
                )
        """

        example = {}
        example["document_id"] = txt_file.with_suffix("").name
        with txt_file.open() as f:
            example["text"] = f.read()
        a1_file = txt_file.with_suffix(".a1")
        a2_file = txt_file.with_suffix(".a2")
        ann_file = txt_file.with_suffix(".ann")

        ann_lines = []

        if a1_file.exists():
            with a1_file.open() as f:
                ann_lines.extend(f.readlines())

        if a2_file.exists():
            with a2_file.open() as f:
                ann_lines.extend(f.readlines())

        if ann_file.exists():
            with ann_file.open() as f:
                ann_lines.extend(f.readlines())

        example["text_bound_annotations"] = []
        example["events"] = []
        example["relations"] = []
        example["equivalences"] = []
        example["attributes"] = []
        example["normalizations"] = []
        for line in ann_lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("T"):  # Text bound
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["text"] = [fields[2]]
                ann["type"] = fields[1].split()[0]
                ann["offsets"] = []
                span_str = self.remove_prefix(fields[1], (ann["type"] + " "))
                for span in span_str.split(";"):
                    start, end = span.split()
                    ann["offsets"].append([int(start), int(end)])

                example["text_bound_annotations"].append(ann)

            elif line.startswith("E"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                ann["type"], ann["trigger"] = fields[1].split()[0].split(":")

                ann["arguments"] = []
                for role_ref_id in fields[1].split()[1:]:
                    argument = {
                        "role": (role_ref_id.split(":"))[0],
                        "ref_id": (role_ref_id.split(":"))[1],
                    }
                    ann["arguments"].append(argument)

                example["events"].append(ann)

            elif line.startswith("R"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["type"] = fields[1].split()[0]

                ann["head"] = {
                    "role": fields[1].split()[1].split(":")[0],
                    "ref_id": fields[1].split()[1].split(":")[1],
                }
                ann["tail"] = {
                    "role": fields[1].split()[2].split(":")[0],
                    "ref_id": fields[1].split()[2].split(":")[1],
                }

                example["relations"].append(ann)

            # '*' seems to be the legacy way to mark equivalences,
            # but I couldn't find any info on the current way
            # this might have to be adapted dependent on the brat version
            # of the annotation
            elif line.startswith("*"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["ref_ids"] = fields[1].split()[1:]

                example["equivalences"].append(ann)

            elif line.startswith("A") or line.startswith("M"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                info = fields[1].split()
                ann["type"] = info[0]
                ann["ref_id"] = info[1]

                if len(info) > 2:
                    ann["value"] = info[2]
                else:
                    ann["value"] = ""

                example["attributes"].append(ann)

            elif line.startswith("N"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["text"] = fields[2]

                info = fields[1].split()

                ann["type"] = info[0]
                ann["ref_id"] = info[1]
                ann["resource_name"] = info[2].split(":")[0]
                ann["cuid"] = info[2].split(":")[1]
                example["normalizations"].append(ann)

        return example

    def brat_parse_to_bigbio_kb(self, brat_parse: Dict, entity_types: Iterable[str]) -> Dict:
        """
        Transform a brat parse (conforming to the standard brat schema) obtained with
        `parse_brat_file` into a dictionary conforming to the `bigbio-kb` schema (as defined in ../schemas/kb.py)

        :param brat_parse:
        :param entity_types: Entity types of the dataset. This should include all types of `T` annotations that are not event triggers and will be different in different datasets.
        """

        unified_example = {}

        # Prefix all ids with document id to ensure global uniqueness,
        # because brat ids are only unique within their document
        id_prefix = brat_parse["document_id"] + "_"

        # identical
        unified_example["document_id"] = brat_parse["document_id"]
        unified_example["passages"] = [
            {
                "id": id_prefix + "_text",
                "type": "abstract",
                "text": [brat_parse["text"]],
                "offsets": [[0, len(brat_parse["text"])]],
            }
        ]

        # get normalizations
        ref_id_to_normalizations = defaultdict(list)
        for normalization in brat_parse["normalizations"]:
            ref_id_to_normalizations[normalization["ref_id"]].append(
                {
                    "db_name": normalization["resource_name"],
                    "db_id": normalization["cuid"],
                }
            )

        # separate entities and event triggers
        unified_example["entities"] = []
        id_to_event_trigger = {}
        for ann in brat_parse["text_bound_annotations"]:
            if ann["type"] in entity_types:
                entity_ann = ann.copy()
                entity_ann["id"] = id_prefix + entity_ann["id"]
                entity_ann["normalized"] = ref_id_to_normalizations[ann["id"]]
                unified_example["entities"].append(entity_ann)
            else:
                id_to_event_trigger[ann["id"]] = ann

        unified_example["events"] = []
        for event in brat_parse["events"]:
            event = event.copy()
            event["id"] = id_prefix + event["id"]
            trigger = id_to_event_trigger[event["trigger"]]
            event["trigger"] = {
                "text": trigger["text"].copy(),
                "offsets": trigger["offsets"].copy(),
            }
            for argument in event["arguments"]:
                argument["ref_id"] = id_prefix + argument["ref_id"]

            unified_example["events"].append(event)

        # massage relations
        unified_example["relations"] = []
        for ann in brat_parse["relations"]:
            unified_example["relations"].append(
                {
                    "arg1_id": id_prefix + ann["head"]["ref_id"],
                    "arg2_id": id_prefix + ann["tail"]["ref_id"],
                    "id": id_prefix + ann["id"],
                    "type": ann["type"],
                    "normalized": [],
                }
            )

        # get coreferences
        unified_example["coreferences"] = []
        for i, ann in enumerate(brat_parse["equivalences"], start=1):
            is_entity_cluster = True
            for ref_id in ann["ref_ids"]:
                if not ref_id.startswith("T"):  # not textbound -> no entity
                    is_entity_cluster = False
                elif ref_id in id_to_event_trigger:  # event trigger -> no entity
                    is_entity_cluster = False
            if is_entity_cluster:
                entity_ids = [id_prefix + i for i in ann["ref_ids"]]
                unified_example["coreferences"].append(
                    {"id": id_prefix + str(i), "entity_ids": entity_ids}
                )

        return unified_example
