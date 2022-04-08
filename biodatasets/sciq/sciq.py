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

import os
import json
import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks


_DATASETNAME = "sciq"

_CITATION = """
@inproceedings{welbl-etal-2017-crowdsourcing,
    title = "Crowdsourcing Multiple Choice Science Questions",
    author = "Welbl, Johannes  and
      Liu, Nelson F.  and
      Gardner, Matt",
    booktitle = "Proceedings of the 3rd Workshop on Noisy User-generated Text",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-4413",
    doi = "10.18653/v1/W17-4413",
    pages = "94--106",
}
"""

_DESCRIPTION = """
The SciQ dataset contains 13,679 crowdsourced science exam questions about Physics, Chemistry and Biology, \
among others. The questions are in multiple-choice format with 4 answer options each. \
For most questions, an additional paragraph with supporting evidence for the correct answer is provided.
"""

_HOMEPAGE = "https://allenai.org/data/sciq"

_LICENSE = "CC BY-NC 3.0"

_URLs = "https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip"

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class SciQ(datasets.GeneratorBasedBuilder):
    """SciQ: a crowdsourced science exam QA dataset."""

    DEFAULT_CONFIG_NAME = f"sciq_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        # Source schema
        BigBioConfig(
            name="sciq_source",
            version=SOURCE_VERSION,
            description="SciQ source schema",
            schema="source",
            subset_id=f"sciq",
        ),
        # BigBio schema: question answering
        BigBioConfig(
            name=f"sciq_bigbio_qa",
            version=BIGBIO_VERSION,
            description="SciQ simplified BigBio schema",
            schema="bigbio_qa",
            subset_id="sciq",
        ),
    ]

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "distractor1": datasets.Value("string"),
                    "distractor2": datasets.Value("string"),
                    "distractor3": datasets.Value("string"),
                    "correct_answer": datasets.Value("string"),
                    "support": datasets.Value("string"),
                }
            )
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
        dl_dir = dl_manager.download_and_extract(_URLs)
        data_dir = os.path.join(dl_dir, "SciQ dataset-2 3")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.json"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "valid.json"),
                    "split": datasets.Split.VALIDATION,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.json"),
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if self.config.schema == "source":
            for i, d in enumerate(data):
                yield i, d
        elif self.config.schema == "bigbio_qa":
            question_type = "multiple_choice"
            for i, d in enumerate(data):
                id = f"{split}_{i}"
                yield id, {
                    "id": id,
                    "question_id": id,
                    "document_id": id,
                    "question": d["question"],
                    "type": question_type,
                    "context": d["support"],
                    "answer": [d["correct_answer"]],
                    "choices": [
                        d["distractor1"],
                        d["distractor2"],
                        d["distractor3"],
                        d["correct_answer"],
                    ],
                }
