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

# TODO: FIXME: Add a description
"""
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

[bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""

import ijson
import json
from pathlib import Path
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{tsatsaronis2015overview,
    title        = {
        An overview of the BIOASQ large-scale biomedical semantic indexing and
        question answering competition
    },
    author       = {
        Tsatsaronis, George and Balikas, Georgios and Malakasiotis, Prodromos
        and Partalas, Ioannis and Zschunke, Matthias and Alvers, Michael R and
        Weissenborn, Dirk and Krithara, Anastasia and Petridis, Sergios and
        Polychronopoulos, Dimitris and others
    },
    year         = 2015,
    journal      = {BMC bioinformatics},
    publisher    = {BioMed Central Ltd},
    volume       = 16,
    number       = 1,
    pages        = 138
}
"""

_DATASETNAME = "bioasq_task_a"

# TODO: Find description and copy it
_BIOASQ_2014A_DESCRIPTION = ""
_BIOASQ_2014bA_DESCRIPTION = ""

_BIOASQ_2015A_DESCRIPTION = ""
_BIOASQ_2015bA_DESCRIPTION = ""

_BIOASQ_2016A_DESCRIPTION = ""
_BIOASQ_2016bA_DESCRIPTION = ""

_BIOASQ_2017A_DESCRIPTION = ""
_BIOASQ_2018A_DESCRIPTION = ""
_BIOASQ_2019A_DESCRIPTION = ""
_BIOASQ_2020A_DESCRIPTION = ""
_BIOASQ_2021A_DESCRIPTION = ""
_BIOASQ_2022A_DESCRIPTION = ""

_DESCRIPTION = {
    "bioasq_2014a": _BIOASQ_2014A_DESCRIPTION,
    "bioasq_2014ba": _BIOASQ_2014bA_DESCRIPTION,
    "bioasq_2015a": _BIOASQ_2015A_DESCRIPTION,
    "bioasq_2015ba": _BIOASQ_2015bA_DESCRIPTION,
    "bioasq_2016a": _BIOASQ_2016A_DESCRIPTION,
    "bioasq_2016ba": _BIOASQ_2016bA_DESCRIPTION,
    "bioasq_2017a": _BIOASQ_2017A_DESCRIPTION,
    "bioasq_2018a": _BIOASQ_2018A_DESCRIPTION,
    "bioasq_2019a": _BIOASQ_2019A_DESCRIPTION,
    "bioasq_2020a": _BIOASQ_2020A_DESCRIPTION,
    "bioasq_2021a": _BIOASQ_2021A_DESCRIPTION,
    "bioasq_2022a": _BIOASQ_2022A_DESCRIPTION,
}

_HOMEPAGE = "http://participants-area.bioasq.org/datasets/"

# Data access requires prior registration with BioASQ.
# See http://participants-area.bioasq.org/accounts/register/
_LICENSE = "https://www.nlm.nih.gov/databases/download/terms_and_conditions.html"

# TODO: FIXME: Add bioasq 2013
_URLS = {
    "bioasq_2014a": "allMeSH.zip",
    "bioasq_2014ba": "allMeSH_limitjournals.zip",
    "bioasq_2015a": "allMeSH.zip",
    "bioasq_2015ba": "allMeSH_limitjournals.zip",
    "bioasq_2016a": "allMeSH_2016.zip",
    "bioasq_2016ba": "allMeSH_limitjournals_2016.zip",
    "bioasq_2017a": "allMeSH_2017.zip",
    "bioasq_2018a": "allMeSH_2018.zip",
    "bioasq_2019a": "allMeSH_2019.zip",
    "bioasq_2020a": "allMeSH_2020.zip",
    "bioasq_2021a": "allMeSH_2021.zip",
    "bioasq_2022a": "allMeSH_2022.zip",
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BioasqTaskADataset(datasets.GeneratorBasedBuilder):
    """
    BioASQ Task A On Biomedical Text Classification.
    Creates configs for BioASQ A 2013 through BioASQ A 2021.
    """

    DEFAULT_CONFIG_NAME = "bioasq_2014a_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # BioASQ A 2014 through BioASQ A 2022
    BUILDER_CONFIGS = []
    for year in ((
        "2014",
        "2014b",
        "2015",
        "2015b",
        "2016",
        "2016b",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
    )):
        BUILDER_CONFIGS.extend([
            BigBioConfig(
                name=f"bioasq_{year}a_source",
                version=SOURCE_VERSION,
                description=f"bioasq {year} Task A source schema",
                schema="source",
                subset_id=f"bioasq_{year}a",
            ),
            BigBioConfig(
                name=f"bioasq_{year}a_bigbio_text",
                version=BIGBIO_VERSION,
                description=f"bioasq {year} Task A in simplified BigBio schema",
                schema="bigbio_text",
                subset_id=f"bioasq_{year}a",
            )
        ])

    def _info(self) -> datasets.DatasetInfo:
        # BioASQ Task A source schema
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "abstractText": datasets.Value("string"),
                    "journal": datasets.Value("string"),
                    "meshMajor": [datasets.Value("string")],
                    "pmid": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "year": datasets.Value("string"),
                }
            )
        # simplified schema for text classification tasks
        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION[self.config.subset_id],
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")

        data_dir = self.config.data_dir
        url = _URLS[self.config.subset_id]

        train_data_dir = dl_manager.download_and_extract(Path(data_dir) / url)

        subset_filepaths = {
            "bioasq_2014a": "allMeSH.json",
            "bioasq_2014ba": "allMeSH_limitjournals.json",
            "bioasq_2015a": "allMeSH.json",
            "bioasq_2015ba": "allMeSH_limitjournals.json",
            "bioasq_2016a": "allMeSH_2016.json",
            "bioasq_2016ba": "allMeSH_limitjournals_2016.json",
            "bioasq_2017a": "allMeSH_2017.json",
            "bioasq_2018a": "allMeSH_2018.json",
            "bioasq_2019a": "allMeSH_2019.json",
            "bioasq_2020a": "allMeSH_2020.json",
            "bioasq_2021a": "allMeSH_2021.json",
            "bioasq_2022a": "allMeSH_2022.json",
        }
        filepath = Path(train_data_dir) / subset_filepaths[self.config.subset_id]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            ),
        ]

    def _generate_articles(self, filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            if self.config.subset_id in ("bioasq_2014a", "bioasq_2014ba", "bioasq_2015a", "bioasq_2015ba"):
                article_index = 0

                for line in f:
                    try:
                        record = json.loads(line.rstrip(",\n"))
                    except json.decoder.JSONDecodeError:
                        # TODO: FIXME: Nicer handling of these lines
                        if "'articles'" in line:
                            continue
                        else:
                            print("FAILED:", line)
                            continue
                    else:
                        yield article_index, record
                        article_index += 1
            else:
                for article_index, record in enumerate(ijson.items(f, "articles.item")):
                    yield article_index, record

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        for record_index, record in self._generate_articles(filepath=filepath):
            if self.config.schema == "source":
                yield record_index, record
            elif self.config.schema == "bigbio_text":
                yield record_index, {
                    "id": record["pmid"],
                    "document_id": record["title"],
                    "text": record["abstractText"],
                    "labels": record["meshMajor"],
                }
