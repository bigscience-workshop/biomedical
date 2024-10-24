# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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

"""
A dataset loading script for the CANTEMIST corpus.

The CANTEMIST datset is collection of 1301 oncological clinical case reports 
written in Spanish, with tumor morphology mentions manually annotated and 
mapped by clinical experts to a controlled terminology. Every tumor morphology 
mention is linked to an eCIE-O code (the Spanish equivalent of ICD-O).
"""

import csv
import os.path
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import kb_features
from .bigbiohub import text_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks
from .bigbiohub import parse_brat_file
from .bigbiohub import brat_parse_to_bigbio_kb


_LANGUAGES = ["Spanish"]
_PUBMED = False
_LOCAL = False
_CITATION = """\
@article{miranda2020named,
  title={Named Entity Recognition, Concept Normalization and Clinical Coding: Overview of the Cantemist Track for Cancer Text Mining in Spanish, Corpus, Guidelines, Methods and Results.},
  author={Miranda-Escalada, Antonio and Farr{\'e}, Eul{\`a}lia and Krallinger, Martin},
  journal={IberLEF@ SEPLN},
  pages={303--323},
  year={2020}
}
"""

_DATASETNAME = "cantemist"
_DISPLAYNAME = "CANTEMIST"

_DESCRIPTION = """\
Collection of 1301 oncological clinical case reports written in Spanish, with tumor morphology mentions \
manually annotated and mapped by clinical experts to a controlled terminology. Every tumor morphology \
mention is linked to an eCIE-O code (the Spanish equivalent of ICD-O).

The original dataset is distributed in Brat format, and was randomly sampled into 3 subsets. \
The training, development and test sets contain 501, 500 and 300 documents each, respectively.

This dataset was designed for the CANcer TExt Mining Shared Task, sponsored by Plan-TL. \
The task is divided in 3 subtasks: CANTEMIST-NER, CANTEMIST_NORM and CANTEMIST-CODING.

CANTEMIST-NER track: requires finding automatically tumor morphology mentions. All tumor morphology \
mentions are defined by their corresponding character offsets in UTF-8 plain text medical documents. 

CANTEMIST-NORM track: clinical concept normalization or named entity normalization task that requires \
to return all tumor morphology entity mentions together with their corresponding eCIE-O-3.1 codes \
i.e. finding and normalizing tumor morphology mentions.

CANTEMIST-CODING track: requires returning for each of document a ranked list of its corresponding ICD-O-3 \
codes. This it is essentially a sort of indexing or multi-label classification task or oncology clinical coding. 

For further information, please visit https://temu.bsc.es/cantemist or send an email to encargo-pln-life@bsc.es
"""

_HOMEPAGE = "https://temu.bsc.es/cantemist/?p=4338"

_LICENSE = "Creative Commons Attribution 4.0 International"

_URLS = {
    _DATASETNAME: "data/cantemist.zip",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
    Tasks.TEXT_CLASSIFICATION,
]

_SOURCE_VERSION = "1.6.0"

_BIGBIO_VERSION = "1.0.0"


class CantemistDataset(datasets.GeneratorBasedBuilder):
    """Manually annotated collection of oncological clinical case reports written in Spanish."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="cantemist_source",
            version=SOURCE_VERSION,
            description="CANTEMIST source schema",
            schema="source",
            subset_id="cantemist",
        ),
        BigBioConfig(
            name="cantemist_bigbio_kb",
            version=BIGBIO_VERSION,
            description="CANTEMIST BigBio schema for the NER and NED tasks",
            schema="bigbio_kb",
            subset_id="subtracks_1_2",
        ),
        BigBioConfig(
            name="cantemist_bigbio_text",
            version=BIGBIO_VERSION,
            description="CANTEMIST BigBio schema for the CODING task",
            schema="bigbio_text",
            subset_id="subtrack_3",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cantemist_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "labels": [datasets.Value("string")],  # subtrack 3 codes
                    "text_bound_annotations": [  # T line in brat
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "events": [  # E line in brat
                        {
                            "trigger": datasets.Value("string"),
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
                            "resource_name": datasets.Value("string"),
                            "cuid": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "notes": [  # # lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                },
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        elif self.config.schema == "bigbio_text":
            features = text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """
        Downloads/extracts the data to generate the train, validation and test splits.

        Each split is created by instantiating a `datasets.SplitGenerator`, which will
        call `this._generate_examples` with the keyword arguments in `gen_kwargs`.
        """

        data_dir = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": {
                        "task1": dl_manager.iter_files(
                            os.path.join(data_dir, "train-set", "cantemist-ner")
                        ),
                        "task2": dl_manager.iter_files(
                            os.path.join(data_dir, "train-set", "cantemist-norm")
                        ),
                        "task3": [
                            os.path.join(
                                data_dir,
                                "train-set",
                                "cantemist-coding",
                                "train-coding.tsv",
                            )
                        ],
                    },
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": {
                        "task1": dl_manager.iter_files(
                            os.path.join(data_dir, "test-set", "cantemist-ner")
                        ),
                        "task2": dl_manager.iter_files(
                            os.path.join(data_dir, "test-set", "cantemist-norm")
                        ),
                        "task3": [
                            os.path.join(
                                data_dir,
                                "test-set",
                                "cantemist-coding",
                                "test-coding.tsv",
                            )
                        ],
                    },
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": {
                        "task1": chain(
                            dl_manager.iter_files(
                                os.path.join(data_dir, "dev-set1", "cantemist-ner")
                            ),
                            dl_manager.iter_files(
                                os.path.join(data_dir, "dev-set2", "cantemist-ner")
                            ),
                        ),
                        "task2": chain(
                            dl_manager.iter_files(
                                os.path.join(data_dir, "dev-set1", "cantemist-norm")
                            ),
                            dl_manager.iter_files(
                                os.path.join(data_dir, "dev-set2", "cantemist-norm")
                            ),
                        ),
                        "task3": [
                            os.path.join(
                                data_dir,
                                "dev-set1",
                                "cantemist-coding",
                                "dev1-coding.tsv",
                            ),
                            os.path.join(
                                data_dir,
                                "dev-set2",
                                "cantemist-coding",
                                "dev2-coding.tsv",
                            ),
                        ],
                    },
                },
            ),
        ]

    def _generate_examples(self, filepaths) -> Tuple[int, Dict]:
        """
        This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        Method parameters are unpacked from `gen_kwargs` as given in `_split_generators`.
        """

        if self.config.schema == "source" or self.config.schema == "bigbio_text":
            task3_dict = defaultdict(list)
            for file_path in filepaths["task3"]:
                with open(file_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        task3_dict[row["file"]].append(row["code"])

        if self.config.schema == "source":
            for guid, file_path in enumerate(filepaths["task2"]):
                if os.path.splitext(file_path)[-1] != ".txt":
                    continue
                example = parse_brat_file(
                    Path(file_path), annotation_file_suffixes=[".ann"], parse_notes=True
                )
                # consider few cases where subtrack 3 has no codes for the current document
                example["labels"] = task3_dict.get(example["document_id"], [])
                example["id"] = str(guid)
                yield guid, example

        elif self.config.schema == "bigbio_kb":
            for guid, file_path in enumerate(filepaths["task2"]):
                if os.path.splitext(file_path)[-1] != ".txt":
                    continue
                parsed_brat = parse_brat_file(
                    Path(file_path), annotation_file_suffixes=[".ann"], parse_notes=True
                )
                example = brat_parse_to_bigbio_kb(parsed_brat)
                example["id"] = str(guid)
                for i in range(0, len(example["entities"])):
                    normalized_dict = {
                        "db_id": parsed_brat["notes"][i]["text"],
                        "db_name": "eCIE-O-3.1",
                    }
                    example["entities"][i]["normalized"].append(normalized_dict)
                yield guid, example

        elif self.config.schema == "bigbio_text":
            for guid, file_path in enumerate(filepaths["task1"]):
                if os.path.splitext(file_path)[-1] != ".txt":
                    continue
                parsed_brat = parse_brat_file(
                    Path(file_path),
                    annotation_file_suffixes=[".ann"],
                    parse_notes=False,
                )
                # consider few cases where subtrack 3 has no codes for the current document
                labels = task3_dict.get(parsed_brat["document_id"], [])
                example = {
                    "id": str(guid),
                    "document_id": parsed_brat["document_id"],
                    "text": parsed_brat["text"],
                    "labels": labels,
                }
                yield guid, example

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
