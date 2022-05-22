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
from typing import List, Tuple, Dict
import pandas as pd
import datasets
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_CITATION = """\
@misc{ask9medicaldata,
  author    = {Khan, Arbaaz},
  title     = {Sentiment Analysis for Medical Drugs},
  year      = {2019},
  url       = {https://www.kaggle.com/datasets/arbazkhan971/analyticvidhyadatasetsentiment},
}
"""

_DATASETNAME = "medical_data"

_DESCRIPTION = """\
    This dataset is designed to do multiclass classification on medical drugs
"""

_HOMEPAGE = ""

_LICENSE = ""

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]


_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MedicaldataDatatset(datasets.GeneratorBasedBuilder):
    """This dataset contains comments about patients and the sentiment in those comments about a specific drug that's mentioned.
    1 - Negative sentiment
    2 - Positive sentiment
    0 - Neutral"""

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
            name=f"{_DATASETNAME}_bigbio_te",
            version=BIGBIO_VERSION,
            description=f"{_DATASETNAME} BigBio schema",
            schema="bigbio_te",
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
                    "sentiment": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_te":
            features = schemas.entailment_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
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

        if self.config.schema == "source":
            csv_reader = pd.read_csv(filepath, dtype="object")
            # Train and test splits handled differently as test split is missing values
            if split == "train":
                for _cols, line in csv_reader.iterrows():
                    document = {}
                    document["hash"] = line["unique_hash"]
                    document["text"] = line["text"]
                    document["drug_name"] = line["drug"]
                    document["sentiment"] = line["sentiment"]
                    yield document["hash"], document
            else:
                for _cols, line in csv_reader.iterrows():
                    document = {}
                    document["hash"] = line["unique_hash"]
                    document["text"] = line["text"]
                    document["drug_name"] = line["drug"]
                    document["sentiment"] = None
                    yield document["hash"], document

        elif self.config.schema == "bigbio_te":
            csv_reader = pd.read_csv(filepath, dtype="object")
            # Train and test splits handled differently as test split is missing values
            if split == "train":
                for _cols, line in csv_reader.iterrows():
                    document = {}
                    document["id"] = line["unique_hash"]
                    document["premise"] = line["text"]
                    document["hypothesis"] = line["drug"]
                    document["label"] = line["sentiment"]
                    yield document["id"], document
            else:
                for _cols, line in csv_reader.iterrows():
                    document = {}
                    document["id"] = line["unique_hash"]
                    document["premise"] = line["text"]
                    document["hypothesis"] = line["drug"]
                    document["label"] = None
                    yield document["id"], document
