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

import os
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LANGUAGES = [Lang.VI]
_LOCAL = False
_CITATION = """@inproceedings{PhoNER_COVID19,
    author    = {Thinh Hung Truong and Mai Hoang Dao and Dat Quoc Nguyen},  
    title     = {COVID-19 Named Entity Recognition for Vietnamese}, 
    booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for 
                 Computational Linguistics: Human Language Technologies},
    year      = {2021},
    pages     = {2146–2153},
    url       = {https://aclanthology.org/2021.naacl-main.173/},
    doi       = {10.18653/v1/2021.naacl-main.173},
"""

_DATASETNAME = "pho_ner"

_DESCRIPTION = """PhoNER_COVID19 is a dataset for recognizing COVID-19 related named entities in Vietnamese, 
consisting of 35K entities over 10K sentences. We define 10 entity types with the aim of extracting key information 
related to COVID-19 patients, which are especially useful in downstream applications. In general, these entity types 
can be used in the context of not only the COVID-19 pandemic but also in other future epidemics """

_HOMEPAGE = "https://github.com/VinAIResearch/PhoNER_COVID19"

_LICENSE = "\
By downloading the PhoNER_COVID19 dataset, USER agrees: \
- to use PhoNER_COVID19 for research or educational purposes only.\
- to not distribute PhoNER_COVID19 or part of PhoNER_COVID19 in any original or modified form.\
- and to cite our NAACL paper above whenever PhoNER_COVID19 is employed to help produce published results."

_URLS = {
    "source": "https://github.com/VinAIResearch/PhoNER_COVID19/archive/refs/heads/master.zip",
    "bigbio_kb": "https://github.com/VinAIResearch/PhoNER_COVID19/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class PhoNerDataset(datasets.GeneratorBasedBuilder):
    """COVID-19 Named Entity Recognition for Vietnamese"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pho_ner_source",
            version=SOURCE_VERSION,
            description="Pho Ner source schema for word",
            schema="source",
            subset_id="pho_ner_word",
        ),
        BigBioConfig(
            name="pho_ner_syllable_source",
            version=SOURCE_VERSION,
            description="Pho Ner source schema for syllable",
            schema="source",
            subset_id="pho_ner_syllable",
        ),
        BigBioConfig(
            name="pho_ner_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Pho Ner BigBio schema for word",
            schema="bigbio_kb",
            subset_id="pho_ner_word",
        ),
        BigBioConfig(
            name="pho_ner_syllable_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Pho Ner BigBio schema for syllable",
            schema="bigbio_kb",
            subset_id="pho_ner_syllable",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pho_ner_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "passages": [
                        {
                            "id": datasets.Value("int32"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "entities": [
                                {
                                    "id": datasets.Value("int32"),
                                    "offsets": datasets.Sequence(
                                        [datasets.Value("int32")]
                                    ),
                                    "text": datasets.Sequence(datasets.Value("string")),
                                    "type": datasets.Value("string"),
                                }
                            ],
                        }
                    ]
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[self.config.schema]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "PhoNER_COVID19-main/data"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "PhoNER_COVID19-main/data"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "PhoNER_COVID19-main/data"),
                    "split": "dev",
                },
            ),
        ]

    @staticmethod
    def join_entity(type_ner: str, current_entity_list: List, entity_list: List):
        entity_text = " ".join(current_entity_list)
        entity_list.append((type_ner, entity_text))
        return entity_list, []

    @staticmethod
    def extract_entity(
        entity_list: List, text: str, current_id: int, schema: str
    ) -> List:
        entities = []
        ignore_index = 0
        for index, (entity_type, entity) in enumerate(entity_list):
            if entity_type == "O":
                ignore_index += 1
                continue
            else:
                if schema == "source":
                    entities.append(
                        {
                            "id": index + current_id - ignore_index,
                            "offsets": [
                                (text.index(entity), text.index(entity) + len(entity))
                            ],
                            "text": [entity],
                            "type": entity_type,
                        }
                    )
                elif schema == "bigbio_kb":
                    entities.append(
                        {
                            "id": str(index + current_id - ignore_index),
                            "offsets": [
                                (text.index(entity), text.index(entity) + len(entity))
                            ],
                            "text": [entity],
                            "type": entity_type,
                            "normalized": [],
                        }
                    )
        return entities

    def _generate_original_examples(self, contents, schema):
        """Generates examples from source files."""
        id = 0
        uid = 0
        passages_word = []
        entity_list = []
        type_ner = None
        current_entity_list = []
        start_sentence = 1
        for idx, content in enumerate(contents):
            data = {}
            line = content.replace("\n", "")
            if len(line) > 0:
                word, entity = line.split(" ")
                passages_word.append(word)
                # Get first value of entity
                if type_ner is None:
                    type_ner = entity.split("-")[-1]
                # check whether current entity is changed
                elif type_ner != entity.split("-")[-1]:
                    entity_list, current_entity_list = self.join_entity(
                        type_ner, current_entity_list, entity_list
                    )
                    type_ner = entity.split("-")[-1]
                current_entity_list.append(word)

            else:
                # join passage word
                text = " ".join(passages_word)
                entity_list, current_entity_list = self.join_entity(
                    type_ner, current_entity_list, entity_list
                )
                type_ner = None
                if schema == "source":
                    entities = self.extract_entity(
                        entity_list, text, id, schema="source"
                    )
                    data = {
                        "passages": [
                            {
                                "id": uid,
                                "text": [text],
                                "offsets": [(start_sentence, idx)],
                                "entities": entities,
                            }
                        ],
                    }
                    id += len(entities)
                    start_sentence = idx + 2

                elif schema == "bigbio_kb":
                    passages_id = id + 1
                    entities = self.extract_entity(
                        entity_list,
                        text,
                        current_id=passages_id + 1,
                        schema="bigbio_kb",
                    )
                    data = {
                        "id": str(id),
                        "document_id": [],
                        "passages": [
                            {
                                "id": str(passages_id),
                                "type": "",
                                "text": [text],
                                "offsets": [(0, len(text))],
                            }
                        ],
                        "entities": entities,
                        "relations": [],
                        "events": [],
                        "coreferences": [],
                    }
                    id += len(entities) + 2

                current_idx = uid
                uid += 1
                entity_list = []
                passages_word = []
                yield current_idx, data

    def _generate_examples(self, filepath: str, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            if self.config.subset_id == "pho_ner_word":
                with open(
                    os.path.join(filepath, f"word/{split}_word.conll"), encoding="utf-8"
                ) as f:
                    contents = f.readlines()
            elif self.config.subset_id == "pho_ner_syllable":
                with open(
                    os.path.join(filepath, f"syllable/{split}_syllable.conll"),
                    encoding="utf-8",
                ) as f:
                    contents = f.readlines()
            genny = self._generate_original_examples(
                contents=contents, schema=self.config.schema
            )
            for _id, sample in genny:
                yield _id, sample
        elif self.config.schema == "bigbio_kb":
            if self.config.subset_id == "pho_ner_word":
                with open(
                    os.path.join(filepath, f"word/{split}_word.conll"), encoding="utf-8"
                ) as f:
                    contents = f.readlines()
            elif self.config.subset_id == "pho_ner_syllable":
                with open(
                    os.path.join(filepath, f"syllable/{split}_syllable.conll"),
                    encoding="utf-8",
                ) as f:
                    contents = f.readlines()
            genny = self._generate_original_examples(
                contents=contents, schema=self.config.schema
            )
            for _id, sample in genny:
                yield _id, sample
