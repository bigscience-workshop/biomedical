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
StudyTypeTeller dataset consists of 2,645 PubMed titles and abstracts, manually annotated for 14 biomedical study types. These include randomized controlled trials (RCTs), different subtypes of human and animal studies, systematic and non-systematic reviews, study protocols, case reports, and in vitro studies.
Each record contains the title and abstract text along with a single study type label, enabling supervised classification tasks in the context of biomedical literature curation, especially systematic review workflows.
"""

import json
import os
from typing import Dict, List, Tuple

import datasets
from .bigbiohub import BigBioConfig, Tasks, text_features

_LANGUAGES = ["English"]
_PUBMED = True
_LOCAL = False

_CITATION = """\
@article{doneva2025studytypeteller,
  author    = {Doneva, Emilova S and de Viragh, S and Hubarava, H and Schandelmaier, S and Briel, M and Ineichen, BV},
  title     = {StudyTypeTeller—Large language models to automatically classify research study types for systematic reviews},
  journal   = {Research Synthesis Methods},
  volume    = {16},
  number    = {6},
  pages     = {1005--1024},
  year      = {2025},
  doi       = {10.1017/rsm.2025.10031}
}
"""

_DATASETNAME = "studytypeteller"
_DISPLAYNAME = "StudyTypeTeller"

_DESCRIPTION = """\
The dataset consists of 2,645 PubMed titles and abstracts, manually annotated for study type classification in the biomedical domain. Each record is labeled according to one of 14 predefined study types.

The annotated categories include randomized controlled trials (RCTs), different subtypes of RCTs, animal studies, systematic reviews, study protocols, case reports, and in vitro studies. The annotations were created by human experts.

This corpus is designed to support automated classification of study types and to facilitate research on reducing the workload associated with abstract screening in systematic reviews.
"""


_HOMEPAGE = "https://github.com/Ineichen-Group/StudyTypeTeller"

_LICENSE = "CC0_1p0"


_URL = "https://raw.githubusercontent.com/Ineichen-Group/StudyTypeTeller/main/data/data_splits_stratified/6-2-2_all_classes_enriched/"
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
    "test": _URL + "test.json",
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

_CLASS_NAMES = [
    "Clinical-study-protocol",
    "Human-systematic-review",
    "Non-systematic-review",
    "Human-RCT-non-drug-intervention",
    "Human-RCT-drug-intervention",
    "Human-RCT-non-intervention",
    "Human-case-report",
    "Human-non-RCT-non-drug-intervention",
    "Human-non-RCT-drug-intervention",
    "Animal-systematic-review",
    "Animal-non-drug-intervention",
    "Animal-drug-intervention",
    "Animal-other",
    "In-vitro-study",
    "Remaining",
]


class StudyTypeTellerDataset(datasets.GeneratorBasedBuilder):
    """The dataset consists of 2,645 PubMed titles and abstracts, manually annotated for 14 biomedical study types, including randomized controlled trials, animal studies, systematic reviews, protocols, case reports, and in vitro studies."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="studytypeteller_source",
            version=SOURCE_VERSION,
            description="studytypeteller source schema",
            schema="source",
            subset_id="studytypeteller",
        ),
        BigBioConfig(
            name="studytypeteller_bigbio_text",
            version=BIGBIO_VERSION,
            description="studytypeteller BigBio schema",
            schema="bigbio_text",
            subset_id="studytypeteller",
        ),
    ]

    DEFAULT_CONFIG_NAME = "studytypeteller_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "label": [datasets.ClassLabel(names=_CLASS_NAMES)],
                }
            )

        elif self.config.schema == "bigbio_text":
            features = text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                    "split": "dev",
                },
            ),
        ]

    @staticmethod
    def get_source_example(uid, entry):
        pmid = entry.get("document_id", "").strip()
        title = entry.get("title", "").strip()
        abstract = entry.get("abstract", "").strip()
        label = entry.get("label", "").strip()
        text = title + "|" + abstract

        doc = {
            "document_id": pmid,
            "text": text,
            "title": title,
            "abstract": abstract,
            "label": [label],
        }

        return uid, doc

    @staticmethod
    def get_bigbio_example(uid, entry):
        pmid = entry.get("document_id", "").strip()
        title = entry.get("title", "").strip()
        abstract = entry.get("abstract", "").strip()
        label = entry.get("label", "").strip()
        text = title + "|" + abstract

        # Build the final document structure
        doc = {
            "id": uid,
            "document_id": pmid,
            "text": text,
            "labels": [label],
        }

        return uid, doc

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r") as f:
            data = json.load(f)
            uid = 0
            # Iterate over each entry in the JSON file
            for entry in data:
                if self.config.schema == "source":
                    yield self.get_source_example(uid, entry)
                elif self.config.schema == "bigbio_text":
                    yield self.get_bigbio_example(uid, entry)
                uid += 1
