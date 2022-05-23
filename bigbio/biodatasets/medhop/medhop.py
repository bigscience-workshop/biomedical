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
import json

import datasets

from bigbio.utils import schemas
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.configs import BigBioConfig

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@article{welbl-etal-2018-constructing,
title = Constructing Datasets for Multi-hop Reading Comprehension Across Documents,
author = Welbl, Johannes and Stenetorp, Pontus and Riedel, Sebastian,
journal = Transactions of the Association for Computational Linguistics,
volume = 6,
year = 2018,
address = Cambridge, MA,
publisher = MIT Press,
url = https://aclanthology.org/Q18-1021,
doi = 10.1162/tacl_a_00021,
pages = 287--302,
abstract = {
    Most Reading Comprehension methods limit themselves to queries which
    can be answered using a single sentence, paragraph, or document.
    Enabling models to combine disjoint pieces of textual evidence would
    extend the scope of machine comprehension methods, but currently no
    resources exist to train and test this capability. We propose a novel
    task to encourage the development of models for text understanding
    across multiple documents and to investigate the limits of existing
    methods. In our task, a model learns to seek and combine evidence
    -- effectively performing multihop, alias multi-step, inference.
    We devise a methodology to produce datasets for this task, given a
    collection of query-answer pairs and thematically linked documents.
    Two datasets from different domains are induced, and we identify
    potential pitfalls and devise circumvention strategies. We evaluate
    two previously proposed competitive models and find that one can
    integrate information across documents. However, both models
    struggle to select relevant information; and providing documents
    guaranteed to be relevant greatly improves their performance. While
    the models outperform several strong baselines, their best accuracy
    reaches 54.5 % on an annotated test set, compared to human
    performance at 85.0 %, leaving ample room for improvement.
}
"""

_DESCRIPTION = """\
With the same format as WikiHop, this dataset is based on research paper
abstracts from PubMed, and the queries are about interactions between
pairs of drugs. The correct answer has to be inferred by combining
information from a chain of reactions of drugs and proteins.
"""

_DATASETNAME = "MedHop"

_HOMEPAGE = "http://qangaroo.cs.ucl.ac.uk/"

_LICENSE = "CC BY-SA 3.0"

_BASE_GDRIVE = "https://drive.google.com/uc?export=download&confirm=yTib&id="

_URLs = {
    "source": _BASE_GDRIVE + "1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA",
    "bigbio_qa": _BASE_GDRIVE + "1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class MedHopDataset(datasets.GeneratorBasedBuilder):
    """MedHop"""

    DEFAULT_CONFIG_NAME = "medhop_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="medhop_source",
            version=SOURCE_VERSION,
            description="MedHop source schema",
            schema="source",
            subset_id="MedHop",
        ),
        BigBioConfig(
            name="medhop_bigbio_qa",
            version=BIGBIO_VERSION,
            description="MedHop BigBio schema",
            schema="bigbio_qa",
            subset_id="MedHop",
        ),
    ]

    def _info(self):

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "candidates": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                    "supports": datasets.Sequence(datasets.Value("string")),
                    "query": datasets.Value("string"),
                }
            )

        # simplified schema for QA tasks
        elif self.config.schema == "bigbio_qa":

            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        my_urls = _URLs[self.config.schema]
        data_dir = dl_manager.download_and_extract(my_urls)
        data_dir += "/qangaroo_v1.1/medhop/"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.json"),
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":

            with open(filepath, encoding="utf-8") as file:

                uid = 0

                data = json.load(file)

                for i, record in enumerate(data):

                    yield i, {
                        "id": record["id"],
                        "candidates": record["candidates"],
                        "answer": record["answer"],
                        "supports": record["supports"],
                        "query": record["query"],
                    }

                    uid += 1

        elif self.config.schema == "bigbio_qa":

            with open(filepath, encoding="utf-8") as file:

                uid = 0

                data = json.load(file)

                for record in data:

                    record["type"] = "multiple_choice"

                    yield uid, {
                        "id": record["id"],
                        "document_id": record["id"],
                        "question_id": record["id"],
                        "question": record["query"],
                        "type": record["type"],
                        "context": " ".join(record["supports"]),
                        "answer": [record["answer"]],
                        "choices": record["candidates"],
                    }

                    uid += 1
