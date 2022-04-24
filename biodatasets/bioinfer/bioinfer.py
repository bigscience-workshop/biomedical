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
The authors present BioInfer (Bio Information Extraction Resource), a new public resource providing an annotated corpus of biomedical English. We describe an annotation scheme capturing named entities and their relationships along with a dependency analysis of sentence syntax. We further present ontologies defining the types of entities and relationships annotated in the corpus. Currently, the corpus contains 1100 sentences from abstracts of biomedical research articles annotated for relationships, named entities, as well as syntactic dependencies.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import datasets

from biomed_datasets.utils import schemas
from biomed_datasets.utils.configs import BigBioConfig
from biomed_datasets.utils.constants import Tasks

_CITATION = """\
@article{pyysalo2007bioinfer,
  title={BioInfer: a corpus for information extraction in the biomedical domain},
  author={Pyysalo, Sampo and Ginter, Filip and Heimonen, Juho and Bj{\"o}rne, Jari and Boberg, Jorma and J{\"a}rvinen, Jouni and Salakoski, Tapio},
  journal={BMC bioinformatics},
  volume={8},
  number={1},
  pages={1--24},
  year={2007},
  publisher={BioMed Central}
}
"""

_DATASETNAME = "bioinfer"

_DESCRIPTION = """\
A corpus targeted at protein, gene, and RNA relationships which serves as a resource for the development of 
information extraction systems and their components such as parsers and domain analyzers. Currently, the corpus 
contains 1100 sentences from abstracts of biomedical research articles annotated for relationships, named entities, 
as well as syntactic dependencies.
"""

_HOMEPAGE = "https://github.com/metalrt/ppi-dataset"

_LICENSE = "Creative Commons Attribution 2.0 International (CC BY 2.0)"

_URLS = {
    _DATASETNAME: "https://github.com/metalrt/ppi-dataset/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION, Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BioinferDataset(datasets.GeneratorBasedBuilder):
    """1100 sentences from abstracts of biomedical research articles annotated for relationships, named entities, as well as syntactic dependencies."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bioinfer_source",
            version=SOURCE_VERSION,
            description="BioInfer source schema",
            schema="source",
            subset_id="bioinfer",
        ),
        BigBioConfig(
            name="bioinfer_bigbio_kb",
            version=BIGBIO_VERSION,
            description="BioInfer BigBio schema",
            schema="bigbio_kb",
            subset_id="bioinfer",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bioinfer_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "offsets": [[datasets.Value("int32")]],
                            "text": [datasets.Value("string")],
                            "type": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
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
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "ppi-dataset-master/csv_output/BioInfer-train.xml"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "ppi-dataset-master/csv_output/BioInfer-test.xml"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        tree = ET.parse(filepath)
        root = tree.getroot()
        if self.config.schema == "source":
            for guid, sentence in enumerate(root.iter("sentence")):
                example = self._create_example(sentence)
                example["text"] = sentence.attrib["text"]
                example["type"] = "Sentence"
                yield guid, example

        elif self.config.schema == "bigbio_kb":
            for guid, sentence in enumerate(root.iter("sentence")):
                example = self._create_example(sentence)
                example["passages"] = [
                    {
                        "id": f"{sentence.attrib['id']}__text",
                        "type": "Sentence",
                        "text": [sentence.attrib["text"]],
                        "offsets": [(0, len(sentence.attrib["text"]))],
                    }
                ]
                example["events"] = []
                example["coreferences"] = []
                example["id"] = guid
                yield guid, example

    def _create_example(self, sentence):
        example = {}
        example["document_id"] = sentence.attrib["id"]
        example["entities"] = []
        example["relations"] = []
        for tag in sentence:
            if tag.tag == "entity":
                example["entities"].append(self._add_entity(tag))
            elif tag.tag == "interaction":
                example["relations"].append(self._add_interaction(tag))
            else:
                raise ValueError(f"unknown tags: {tag.tag}")
        return example

    @staticmethod
    def _add_entity(entity):
        offsets = [[int(o) for o in offset.split("-")] for offset in entity.attrib["charOffset"].split(",")]
        # For multiple offsets, split entity text accordingly
        if len(offsets) > 1:
            text = []
            i = 0
            for start, end in offsets:
                chunk_len = end - start
                text.append(entity.attrib["text"][i : chunk_len + i])
                i += chunk_len
                while i < len(entity.attrib["text"]) and entity.attrib["text"][i] == " ":
                    i += 1
        else:
            text = [entity.attrib["text"]]
        return {
            "id": entity.attrib["id"],
            "offsets": offsets,
            "text": text,
            "type": entity.attrib["type"],
            "normalized": {},
        }

    @staticmethod
    def _add_interaction(interaction):
        return {
            "id": interaction.attrib["id"],
            "type": interaction.attrib["type"],
            "arg1_id": interaction.attrib["e1"],
            "arg2_id": interaction.attrib["e2"],
            "normalized": {},
        }
