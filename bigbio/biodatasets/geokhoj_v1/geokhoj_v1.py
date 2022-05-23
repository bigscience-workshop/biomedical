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
GEOKhoj v1 contains metadata for 30,000 samples with their respective labels (control/perturbed),
which were labelled using the information available in the metadata.
Metadata has been extracted for samples from Microarray, Transcriptomics
and Single cell experiments which are available on the GEO (Gene Expression Omnibus) database.
"""

import os
from typing import Dict, Tuple

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = "NA"

_DATASETNAME = "geokhoj_v1"

_DESCRIPTION = """\
GEOKhoj v1 is a annotated corpus of control/perturbation labels for 30,000 samples
from Microarray, Transcriptomics and Single cell experiments which are available on
the GEO (Gene Expression Omnibus) database
"""

_HOMEPAGE = "https://github.com/ElucidataInc/GEOKhoj-datasets/tree/main/geokhoj_v1"

_LICENSE = "CC BY-NC 4.0"

_URLS = {
    "source": "https://github.com/ElucidataInc/GEOKhoj-datasets/blob/main/geokhoj_v1/geokhoj_V1.zip?raw=True",
    "bigbio_text": "https://github.com/ElucidataInc/GEOKhoj-datasets/blob/main/geokhoj_v1/geokhoj_V1.zip?raw=True",
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class Geokhojv1Dataset(datasets.GeneratorBasedBuilder):
    """
    GEOKhoj v1 text classification dataset
    """

    DEFAULT_CONFIG_NAME = "geokhoj_v1_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="geokhoj_v1_source",
            version=SOURCE_VERSION,
            description="GEOKhoj v1 source schema",
            schema="source",
            subset_id="geokhoj_v1",
        ),
        BigBioConfig(
            name="geokhoj_v1_bigbio_text",
            version=BIGBIO_VERSION,
            description="GEOKhoj v1 BigBio schema",
            schema="bigbio_text",
            subset_id="geokhoj_v1",
        ),
    ]

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names={0: "control", 1: "perturbation"}),
                    "text": datasets.Value("string"),
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

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls = _URLS[self.config.schema]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "geokhoj_v1/data/train/geo_samples_train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "geokhoj_v1/data/test/geo_samples_test.csv"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, encoding="utf-8", header=None)

        if self.config.schema == "source":
            for id_, row in df.iterrows():
                yield id_, {
                    "id": row[0], 
                    "label": row[1], 
                    "text": row[2]
                }

        elif self.config.schema == "bigbio_text":
            for id_, row in df.iterrows():
                yield id_, {
                    "id": id_ + 1,
                    "document_id": row[0],
                    "text": row[2],
                    "labels": [row[1]],
                }
