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
The Repository for Medical Dataset for Abbreviation Disambiguation for Natural Language Understanding (MeDAL) is
a large medical text dataset curated for abbreviation disambiguation, designed for natural language understanding
pre-training in the medical domain. This script loads the MeDAL dataset in the bigbio KB schema and/or source schema.
"""

import os
from typing import List, Tuple, Dict

import csv 

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{,
    title = {MeDAL\: Medical Abbreviation Disambiguation Dataset for Natural Language Understanding Pretraining},
    author = {Wen, Zhi and Lu, Xing Han and Reddy, Siva},
    booktitle = {Proceedings of the 3rd Clinical Natural Language Processing Workshop},
    month = {Nov},
    year = {2020},
    address = {Online},
    publisher = {Association for Computational Linguistics},
    url = {https://www.aclweb.org/anthology/2020.clinicalnlp-1.15},
    pages = {130--135},
}
"""

_DATASETNAME = "medal"

_DESCRIPTION = """\
The Repository for Medical Dataset for Abbreviation Disambiguation for Natural Language Understanding (MeDAL) is
a large medical text dataset curated for abbreviation disambiguation, designed for natural language understanding
pre-training in the medical domain.
"""

_HOMEPAGE = "https://github.com/BruceWen120/medal"

_LICENSE = """\
The original dataset was retrieved and modified from the NLM website. By using this dataset, you are bound by the terms and conditions specified by NLM:
INTRODUCTION
Downloading data from the National Library of Medicine FTP servers indicates your acceptance of the following Terms and Conditions: No charges, usage fees or royalties are paid to NLM for this data.
MEDLINE/PUBMED SPECIFIC TERMS
NLM freely provides PubMed/MEDLINE data. Please note some PubMed/MEDLINE abstracts may be protected by copyright.
GENERAL TERMS AND CONDITIONS
Users of the data agree to:
- acknowledge NLM as the source of the data by including the phrase "Courtesy of the U.S. National Library of Medicine" in a clear and conspicuous manner,
properly use registration and/or trademark symbols when referring to NLM products, and
- not indicate or imply that NLM has endorsed its products/services/applications.
Users who republish or redistribute the data (services, products or raw data) agree to:
- maintain the most current version of all distributed data, or
- make known in a clear and conspicuous manner that the products/services/applications do not reflect the most current/accurate data available from NLM.
These data are produced with a reasonable standard of care, but NLM makes no warranties express or implied, including no warranty of merchantability or fitness for particular purpose, regarding the accuracy or completeness of the data. Users agree to hold NLM and the U.S. Government harmless from any liability resulting from errors in the data. NLM disclaims any liability for any consequences due to use, misuse, or interpretation of information contained or not contained in the data.
NLM does not provide legal advice regarding copyright, fair use, or other aspects of intellectual property rights. See the NLM Copyright page.
NLM reserves the right to change the type and format of its machine-readable data. NLM will take reasonable steps to inform users of any changes to the format of the data before the data are distributed via the announcement section or subscription to email and RSS updates."""

_URL = "https://zenodo.org/record/4482922/files/"
_URLS = {
    "train": _URL + "train.csv",
    "test": _URL + "test.csv",
    "valid": _URL + "valid.csv",
    "full": _URL + "full_data.csv",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_DISAMBIGUATION] 

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class MedalDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")


    BUILDER_CONFIGS = [
        BigBioConfig(
            name="medal_source",
            version=SOURCE_VERSION,
            description="MeDAL source schema",
            schema="source",
            subset_id="medal",
        ),
        BigBioConfig(
            name="medal_bigbio_kb",
            version=BIGBIO_VERSION,
            description="MeDAL BigBio schema",
            schema="bigbio_kb",
            subset_id="medal",
        ),
    ]

    DEFAULT_CONFIG_NAME = "medal_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features=datasets.Features(
                {
                    "abstract_id": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "location": datasets.Sequence(datasets.Value("int32")),
                    "label": datasets.Sequence(datasets.Value("string"))
                }
            ),

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "bigbio" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument. Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        urls_to_dl = _URLS
        try:
            dl_dir = dl_manager.download_and_extract(urls_to_dl)
        except Exception:
            logger.warning(
                "This dataset is downloaded through Zenodo which is flaky. If this download failed try a few times before reporting an issue"
            )
            raise

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": dl_dir["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": dl_dir["test"], "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": dl_dir["valid"], "split": "val"},
            ),
            datasets.SplitGenerator(
                name="full",
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": dl_dir["full"], "split": "full"},
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files
        with open(filepath, encoding="utf-8") as f:
            data = csv.reader(f)
            # Skip header
            next(data)

            if split == "full":
                id_ = 0
                if self.config.schema == "source":
                    for id_, row in enumerate(data):
                        yield id_, {
                            "abstract_id": -1,
                            "text": row[0],
                            "location": [int(location) for location in row[1].split("|")],
                            "label": [row[2].split("|")]
                        }
                elif self.config.schema == "bigbio_kb":
                       for id_, row in enumerate(data):
                        yield id_, {
                            "id": id_,
                            "document_id": -1,
                            "entities": [ 
                                {
                                    "id": -1,
                                    "type": "abbreviation",
                                    "text": row[0],
                                    "offsets": [int(location) for location in row[1].split("|")],
                                    "normalized": [
                                         { 
                                             "db_name": "medal",
                                             "db_id": [row[2].split("|")],
                                         }
                                    ],
                                }
                            ]
                        }
            else:
                if self.config.schema == "source":
                    for id_, row in enumerate(data):
                        yield id_, {
                            "abstract_id": int(row[0]),
                            "text": row[1],
                            "location": [int(row[2])],
                            "label": [row[3]],
                        }
                elif self.config.schema == "bigbio_kb":
                    for id_, row in enumerate(data):
                        yield id_, {
                                "id": id_,
                                "document_id": int(row[0]),
                                "entities": [
                                    {
                                        "id": -1,
                                        "type": "abbreviation",
                                        "text": row[1],
                                        "offsets": [int(row[2])],
                                        "normalized" : [
                                            {
                                                "db_name": "medal",
                                                "db_id": [row[3]],
                                            }
                                        ] 
                                    }
                                ]
                            }
