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
"""
Medical Question Pairs dataset by McCreery et al (2020) contains pairs of medical questions and paraphrased versions of
the question prepared by medical professional.
"""

import csv
import os
from typing import Dict, Tuple

import datasets
from datasets import load_dataset

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks
from bigbio.utils.license import Licenses

_LOCAL = False
_CITATION = """\
@article{DBLP:journals/biodb/LiSJSWLDMWL16,
  author    = {Krallinger, M., Rabal, O., Lourenço, A.},
  title     = {Effective Transfer Learning for Identifying Similar Questions: Matching User Questions to COVID-19 FAQs},
  journal   = {KDD '20: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  volume    = {3458–3465},
  year      = {2020},
  url       = {https://github.com/curai/medical-question-pair-dataset},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "mqp"

_DESCRIPTION = """\
Medical Question Pairs dataset by McCreery et al (2020) contains pairs of medical questions and paraphrased versions of 
the question prepared by medical professional. Paraphrased versions were labelled as similar (syntactically dissimilar 
but contextually similar ) or dissimilar (syntactically may look similar but contextually dissimilar). Labels 1: similar, 0: dissimilar
"""

_HOMEPAGE = "https://github.com/curai/medical-question-pair-dataset"

_LICENSE = Licenses.UNKNOWN

_URLs = {
    _DATASETNAME: "https://raw.githubusercontent.com/curai/medical-question-pair-dataset/master/mqp.csv",
}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class MQPDataset(datasets.GeneratorBasedBuilder):
    """Medical Question Pairing dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="mqp_source",
            version=SOURCE_VERSION,
            description="MQP source schema",
            schema="source",
            subset_id="mqp",
        ),
        BigBioConfig(
            name="mqp_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="MQP BigBio schema",
            schema="bigbio_pairs",
            subset_id="mqp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mqp_source"

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )

        # Using in pairs schema
        elif self.config.schema == "bigbio_pairs":
            features = schemas.pairs_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""

        if split == "train":  # There's only training dataset available atm
            with open(filepath, encoding="utf-8") as csv_file:
                csv_reader = csv.reader(
                    csv_file,
                    quotechar='"',
                    delimiter=",",
                    quoting=csv.QUOTE_ALL,
                    skipinitialspace=True,
                )

                if self.config.schema == "source":
                    for id_, row in enumerate(csv_reader):
                        document_id, text_1, text_2, label = row
                        yield id_, {
                            "document_id": document_id,
                            "text_1": text_1,
                            "text_2": text_2,
                            "label": label,
                        }

                elif self.config.schema == "bigbio_pairs":
                    # global id (uid) starts from 1
                    uid = 0
                    for id_, row in enumerate(csv_reader):
                        uid += 1
                        document_id, text_1, text_2, label = row
                        yield id_, {
                            "id": uid,  # uid is an unique identifier for every record that starts from 1
                            "document_id": document_id,
                            "text_1": text_1,
                            "text_2": text_2,
                            "label": label,
                        }
        else:
            print("There's no test/val split available for the given dataset")
            return
