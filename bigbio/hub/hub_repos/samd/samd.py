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
import pandas as pd

from .bigbiohub import BigBioConfig, Tasks, pairs_features

_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = True

_CITATION = """\
@misc{ask9medicaldata,
  author    = {Khan, Arbaaz},
  title     = {Sentiment Analysis for Medical Drugs},
  year      = {2019},
  url       = {https://www.kaggle.com/datasets/arbazkhan971/analyticvidhyadatasetsentiment},
}
"""

_DATASETNAME = "samd"
_DISPLAYNAME = "Sentiment Analysis for  Medical Drugs"

_DESCRIPTION = """\
This dataset contains comments about patients and the sentiment in those comments about a specific drug that's \
mentioned.

The dataset has to be download from the Kaggle challenge:
    https://www.kaggle.com/datasets/arbazkhan971/analyticvidhyadatasetsentiment/data
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/arbazkhan971/analyticvidhyadatasetsentiment"
_LICENSE = "UNKNOWN"

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXT_PAIRS_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class SentimentAnalysisMedicalDrugsDatatset(datasets.GeneratorBasedBuilder):
    """This dataset contains comments about patients and the sentiment in those comments about
    a specific drug that's mentioned.

    1 - Negative sentiment
    2 - Positive sentiment
    0 - Neutral
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        BigBioConfig(
            name=f"{_DATASETNAME}_bigbio_pairs",
            version=BIGBIO_VERSION,
            description=f"{_DATASETNAME} BigBio schema",
            schema="bigbio_pairs",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "hash": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "drug_name": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_pairs":
            features = pairs_features
        else:
            raise NotImplementedError(f"Schema {self.config.schema} is not supported")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please download the data from Kaggle abd pass the directory containing "
                "both data files via data_dir kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train_F3WbcTw.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test_tOlRoBf.csv"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        csv_reader = pd.read_csv(filepath, dtype="object")
        for _cols, line in csv_reader.iterrows():
            if self.config.schema == "source":
                document = {
                    "hash": line["unique_hash"],
                    "text": line["text"],
                    "drug_name": line["drug"],
                    "label": line["sentiment"] if split == "train" else None,
                }

                yield document["hash"], document

            elif self.config.schema == "bigbio_pairs":
                document = {
                    "id": line["unique_hash"],
                    "document_id": line["unique_hash"],
                    "text_1": line["text"],
                    "text_2": line["drug"],
                    "label": line["sentiment"] if split == "train" else None,  # test split labels are not given
                }

                yield document["id"], document

            else:
                raise NotImplementedError(f"Schema {self.config.schema} is not supported")
