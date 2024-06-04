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
import re
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas

from .bigbiohub import BigBioConfig, Tasks

_LOCAL = False
_LANGUAGES = ["English"]
_PUBMED = False

_CITATION = """\
@inproceedings{,
  author    = {Dannenfelser, Ruth and Zhong, Jeffrey and Zhang, Ran and Yao, Vicky},
  title     = {Into the Single Cell Multiverse: an End-to-End Dataset for Procedural Knowledge Extraction in Biomedical Texts},
  publisher   = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2024},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2023/file/23e3d86c9a19d0caf2ec997e73dfcfbd-Paper-Datasets_and_Benchmarks.pdf},
}
"""

_DATASETNAME = "flambe"
_DISPLAYNAME = "Flambe"

_DESCRIPTION = """\
FlaMBe is a dataset aimed at procedural knowledge extraction from biomedical texts, 
particularly focusing on single cell research methodologies described in academic papers. It includes 
annotations from 55 full-text articles and 1,195 abstracts, covering nearly 710,000 tokens, and is 
distinguished by its comprehensive named entity recognition (NER) and disambiguation (NED) for 
tissue/cell types, software tools, and computational methods. This dataset, to our knowledge, is 
the largest of its kind for tissue/cell types, links entities to identifiers in relevant knowledge 
bases and annotates nearly 400 workflow relations between tool-context pairs. 
"""

_HOMEPAGE = "https://github.com/ylaboratory/flambe"

_LICENSE = "CC_BY_4p0"

_URLS = {
    _DATASETNAME: "https://zenodo.org/records/10050681/files/data.zip?download",
    "ned": {
        "tissue_test": "https://zenodo.org/records/11218662/files/tissue_ned_test.csv?download",
        "tissue_train": "https://zenodo.org/records/11218662/files/tissue_ned_train.csv?download",
        "tissue_val": "https://zenodo.org/records/11218662/files/tissue_ned_val.csv?download",
        "tool_test": "https://zenodo.org/records/11218662/files/tool_ned_test.csv?download",
        "tool_train": "https://zenodo.org/records/11218662/files/tool_ned_train.csv?download",
        "tool_val": "https://zenodo.org/records/11218662/files/tool_ned_val.csv?download",
    },
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class FlambeDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="flambe_ner_fulltext_tools_source",
            version=SOURCE_VERSION,
            description="NER dataset for tools from full papers",
            schema="source",
            subset_id="flambe_ner_fulltext_tools_source",
        ),
        BigBioConfig(
            name="flambe_ner_fulltext_tissues_source",
            version=SOURCE_VERSION,
            description="NER dataset for tissues from full papers",
            schema="source",
            subset_id="flambe_ner_fulltext_tissues_source",
        ),
        BigBioConfig(
            name="flambe_ner_abstract_tissues_source",
            version=SOURCE_VERSION,
            description="NER dataset for tissues from abstracts",
            schema="source",
            subset_id="flambe_ner_abstract_tissues_source",
        ),
        BigBioConfig(
            name="flambe_ned_tissues",
            version=SOURCE_VERSION,
            description="NED dataset for tissues from full papers",
            schema="source_ned_tissue",
            subset_id="flambe_ned_tissues",
        ),
        BigBioConfig(
            name="flambe_ned_tools",
            version=SOURCE_VERSION,
            description="NED dataset for tools from full papers",
            schema="source_ned_tool",
            subset_id="flambe_ned_tools",
        ),
        BigBioConfig(
            name="flambe_fulltext_tools_bigbio_text",
            version=BIGBIO_VERSION,
            description="Flambe Tissues BigBio schema",
            schema="bigbio_text",
            subset_id="flambe_tool_bigbio",
        ),
        BigBioConfig(
            name="flambe_fulltext_tissues_bigbio_text",
            version=BIGBIO_VERSION,
            description="Flambe Tool BigBio schema",
            schema="bigbio_text",
            subset_id="flambe_tissue_bigbio",
        ),
    ]

    DEFAULT_CONFIG_NAME = "flambe_ner_fulltext_tools_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "tags": datasets.Sequence(datasets.Value("string")),
                }
            )

        elif self.config.schema == "source_ned_tissue":
            features = datasets.Features(
                {
                    "orginal_text": datasets.Value("string"),
                    "mapped_NCIT": datasets.Value("string"),
                    "NCIT_name": datasets.Value("string"),
                }
            )

        elif self.config.schema == "source_ned_tool":
            features = datasets.Features(
                {
                    "orginal_text": datasets.Value("string"),
                    "standardized_name": datasets.Value("string"),
                    "url": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        # TODO: KEEP if your dataset is PUBLIC; remove if not

        # TODO: KEEP if your dataset is PUBLIC; remove if not
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        path = {
            "flambe_ner_fulltext_tools_source": {
                "train": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tools_train.iob"),
                "test": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tools_test.iob"),
                "dev": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tools_validation.iob"),
            },
            "flambe_ner_fulltext_tissues_source": {
                "train": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tissues_train.iob"),
                "test": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tissues_test.iob"),
                "dev": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tissues_validation.iob"),
            },
            "flambe_ner_abstract_tissues_source": {
                "train": os.path.join(data_dir, "data/tags/abstract_iob/abstract_tissues_train.iob"),
                "test": os.path.join(data_dir, "data/tags/abstract_iob/abstract_tissues_test.iob"),
                "dev": os.path.join(data_dir, "data/tags/abstract_iob/abstract_tissues_validation.iob"),
            },
            "flambe_ned_tissues": {
                "train": dl_manager.download_and_extract(_URLS["ned"]["tissue_train"]),
                "test": dl_manager.download_and_extract(_URLS["ned"]["tissue_test"]),
                "dev": dl_manager.download_and_extract(_URLS["ned"]["tissue_val"]),
            },
            "flambe_ned_tools": {
                "train": dl_manager.download_and_extract(_URLS["ned"]["tool_train"]),
                "test": dl_manager.download_and_extract(_URLS["ned"]["tool_test"]),
                "dev": dl_manager.download_and_extract(_URLS["ned"]["tool_val"]),
            },
            "flambe_fulltext_tools_bigbio_text": {
                "train": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tools_train.iob"),
                "test": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tools_test.iob"),
                "dev": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tools_validation.iob"),
            },
            "flambe_fulltext_tissues_bigbio_text": {
                "train": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tissues_train.iob"),
                "test": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tissues_test.iob"),
                "dev": os.path.join(data_dir, "data/tags/fulltext_iob/fulltext_tissues_validation.iob"),
            },
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path[self.config.name]["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": path[self.config.name]["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": path[self.config.name]["dev"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, "r") as f:
                id_value = None
                tokens = []
                tags = []
                key = 0
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts[1] == "begin":
                            if id_value is not None:
                                yield key, {"id": id_value, "tokens": tokens, "tags": tags}
                                key += 1
                                tokens = []
                                tags = []
                            id_value = parts[0]
                        elif parts[1] == "end":
                            yield key, {"id": id_value, "tokens": tokens, "tags": tags}
                            key += 1
                            id_value = None
                            tokens = []
                            tags = []
                        else:
                            tokens.append(parts[0])
                            tags.append(parts[1])
                if id_value is not None:
                    yield key, {"id": id_value, "tokens": tokens, "tags": tags}
                    key += 1
        elif self.config.schema == "bigbio_text":
            with open(filepath, "r") as f:
                id_value = None
                tokens = []
                tags = []
                key = 0
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts[1] == "begin":
                            if id_value is not None:
                                yield key, {
                                    "id": key,
                                    "document_id": id_value,
                                    "text": " ".join(tokens),
                                    "labels": tags,
                                }
                                key += 1
                                tokens = []
                                tags = []
                            id_value = parts[0]
                        elif parts[1] == "end":
                            yield key, {
                                "id": key,
                                "document_id": id_value,
                                "text": " ".join(tokens),
                                "labels": tags,
                            }
                            key += 1
                            id_value = None
                            tokens = []
                            tags = []
                        else:
                            tokens.append(parts[0])
                            tags.append(parts[1])
                if id_value is not None:
                    yield key, {
                        "id": key,
                        "document_id": id_value,
                        "text": " ".join(tokens),
                        "labels": tags,
                    }
                    key += 1

        elif self.config.schema == "source_ned_tissue":
            key = 0
            for line in open(filepath):
                csv_row = line.strip("\n").split(",")
                if csv_row is not None:
                    yield key, {"orginal_text": csv_row[0], "mapped_NCIT": csv_row[1], "NCIT_name": csv_row[2]}
                    key += 1

        elif self.config.schema == "source_ned_tool":
            key = 0
            for line in open(filepath):
                csv_row = line.strip("\n").split(",")
                if csv_row is not None:
                    yield key, {"orginal_text": csv_row[0], "standardized_name": csv_row[1], "url": csv_row[2]}
                    key += 1


if __name__ == "__main__":
    datasets.load_dataset(__file__)
