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
A dataset of 11,832 claims for fact- checking, which are related a range of health topics
including biomedical subjects (e.g., infectious diseases, stem cell research), government healthcare policy
(e.g., abortion, mental health, women’s health), and other public health-related stories
"""

import csv
import os
from pathlib import Path

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

logger = datasets.utils.logging.get_logger(__name__)

_TAGS = []
_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = False
_CITATION = """\
@article{kotonya2020explainable,
  title={Explainable automated fact-checking for public health claims},
  author={Kotonya, Neema and Toni, Francesca},
  journal={arXiv preprint arXiv:2010.09926},
  year={2020}
}
"""

_DATASETNAME = "pubhealth"

_DESCRIPTION = """\
A dataset of 11,832 claims for fact- checking, which are related a range of health topics
including biomedical subjects (e.g., infectious diseases, stem cell research), government healthcare policy
(e.g., abortion, mental health, women’s health), and other public health-related stories
"""

_HOMEPAGE = "https://github.com/neemakot/Health-Fact-Checking/tree/master/data"

_LICENSE = Licenses.MIT

_URLs = {
    _DATASETNAME: "https://drive.google.com/uc?export=download&id=1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj"
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CLASSES = ["true", "false", "unproven", "mixture"]


class PUBHEALTHDataset(datasets.GeneratorBasedBuilder):
    """Pubhealth text classification dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pubhealth_source",
            version=SOURCE_VERSION,
            description="PUBHEALTH source schema",
            schema="source",
            subset_id="pubhealth",
        ),
        BigBioConfig(
            name="pubhealth_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="PUBHEALTH BigBio schema",
            schema="bigbio_pairs",
            subset_id="pubhealth",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pubhealth_source"

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "claim_id": datasets.Value("string"),
                    "claim": datasets.Value("string"),
                    "date_published": datasets.Value("string"),
                    "explanation": datasets.Value("string"),
                    "fact_checkers": datasets.Value("string"),
                    "main_text": datasets.Value("string"),
                    "sources": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=_CLASSES),
                    "subjects": datasets.Value("string"),
                }
            )

        # Using in entailment schema
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
        urls = _URLs[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "PUBHEALTH/train.tsv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "PUBHEALTH/test.tsv"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "PUBHEALTH/dev.tsv"),
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )
            next(csv_reader, None)  # remove column headers
            for id_, row in enumerate(csv_reader):
                # train.tsv/dev.tsv only has 9 columns
                # test.tsv has an additional column at the beginning
                #  Some entries are malformed, will log skipped lines
                if len(row) < 9:
                    logger.warning("Line %s is malformed", id_)
                    continue
                (
                    claim_id,
                    claim,
                    date_published,
                    explanation,
                    fact_checkers,
                    main_text,
                    sources,
                    label,
                    subjects,
                ) = row[
                    -9:
                ]  # only take last 9 columns to fix test.tsv disparity

                if label not in _CLASSES:
                    logger.warning("Line %s is missing label", id_)
                    continue

                if self.config.schema == "source":
                    yield id_, {
                        "claim_id": claim_id,
                        "claim": claim,
                        "date_published": date_published,
                        "explanation": explanation,
                        "fact_checkers": fact_checkers,
                        "main_text": main_text,
                        "sources": sources,
                        "label": label,
                        "subjects": subjects,
                    }

                elif self.config.schema == "bigbio_pairs":
                    yield id_, {
                        "id": id_,  # uid is an unique identifier for every record that starts from 0
                        "document_id": claim_id,
                        "text_1": claim,
                        "text_2": explanation,
                        "label": label,
                    }
