# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and Simon Ott (nomisto)
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
The SciTail dataset is an entailment dataset created from multiple-choice science exams and
web sentences. Each question and the correct answer choice are converted into an assertive
statement to form the hypothesis. We use information retrieval to obtain relevant text from
a large text corpus of web sentences, and use these sentences as a premise P. We crowdsource
the annotation of such premise-hypothesis pair as supports (entails) or not (neutral), in order
to create the SciTail dataset. The dataset contains 27,026 examples with 10,101 examples with
entails label and 16,925 examples with neutral label.
"""

import os
from dataclasses import dataclass

import datasets
import pandas as pd

from utils import schemas

_CITATION = """\
@inproceedings{scitail,
    author = {Tushar Khot and Ashish Sabharwal and Peter Clark},
    booktitle = {AAAI}
    title = {SciTail: A Textual Entailment Dataset from Science Question Answering},
    year = {2018}
}
"""

_DATASETNAME = "scitail"

_DESCRIPTION = """\
The SciTail dataset is an entailment dataset created from multiple-choice science exams and
web sentences. Each question and the correct answer choice are converted into an assertive
statement to form the hypothesis. We use information retrieval to obtain relevant text from
a large text corpus of web sentences, and use these sentences as a premise P. We crowdsource
the annotation of such premise-hypothesis pair as supports (entails) or not (neutral), in order
to create the SciTail dataset. The dataset contains 27,026 examples with 10,101 examples with
entails label and 16,925 examples with neutral label.
"""

_HOMEPAGE = "https://allenai.org/data/scitail"

_LICENSE = "Apache License 2.0"

_URLS = {
    _DATASETNAME: "https://ai2-public-datasets.s3.amazonaws.com/scitail/SciTailV1.1.zip",
}

_SUPPORTED_TASKS = ["TE"]

_SOURCE_VERSION = "1.1.0"

_BIGBIO_VERSION = "1.0.0"


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    name: str = None
    version: str = None
    description: str = None
    schema: str = None
    subset_id: str = None


class SciTail(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="scitail_source",
            version=SOURCE_VERSION,
            description="SciTail source schema",
            schema="source",
            subset_id="scitail",
        ),
        BigBioConfig(
            name="scitail_bigbio_te",
            version=BIGBIO_VERSION,
            description="SciTail BigBio schema",
            schema="bigbio_te",
            subset_id="scitail",
        ),
    ]

    DEFAULT_CONFIG_NAME = "scitail_source"

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.Value("string"),
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

    def _split_generators(self, dl_manager):

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "SciTailV1.1", "tsv_format", "scitail_1.0_train.tsv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "SciTailV1.1", "tsv_format", "scitail_1.0_test.tsv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "SciTailV1.1", "tsv_format", "scitail_1.0_dev.tsv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        # since examples can contain quotes mid text set quoting to QUOTE_NONE (3) when reading tsv
        # e.g.: ... and apply specific "tools" to examples and ... 
        data = pd.read_csv(filepath, sep="\t", names=["premise", "hypothesis", "label"], quoting=3)
        data["id"] = data.index

        if self.config.schema == "source":
            for _, row in data.iterrows():
                yield row["id"], row.to_dict()

        elif self.config.schema == "bigbio_te":
            for _, row in data.iterrows():
                yield row["id"], row.to_dict()
