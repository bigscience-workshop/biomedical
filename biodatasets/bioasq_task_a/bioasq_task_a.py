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
BioASQ Task A On Biomedical Text Classification is based on the standard process
followed by PubMed to index journal abstracts. This task uses PubMed documents,
written in English, along with annotated MeSH terms (by human curators)
that are to be inferred from the documents.

Note that the main difference between datasets from different years, apart from the size,
is the MeSH terms used. For example the 2015 training datasets contain articles
where MeSH 2015 have been assigned. Also, for 2014, 2015 and 2016 there are two
versions of the training data available. The small version (wrt size) consists of
articles that belong to the pool of journals that the BioASQ team used to select the
articles for the test data (this was a subset of the available journals). The bigger
version consists of articles from every available journal. Since 2017 articles for the
test data will be selected from all available journals, so only one corresponding training data
set will be available. The evaluation of the results during each year of the challenge
is performed using the corresponding version of the MeSH terms, thus their usage is highly
recommended. The training datasets of previous years of the challenge are also available
for reference reasons. Note that not every MeSH term is covered in the datasets.

Fore more information about the challenge, the organisers and the relevant
publications please visit: http://bioasq.org/
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

_DESCRIPTION_TEMPLATE = """\
The data are intended to be used as training data for BioASQ 10 A, which will take place during {year}.
There is one file containing the data:
 - {filename}

The training data sets for this task are available for downloading. They
contain annotated articles from PubMed, where annotated means that MeSH terms
have been assigned to the articles by the human curators in PubMed. Table 1
provides information about the provided datasets. Note that the main difference
between those datasets among the different years, apart from the size, is the
MeSH terms used. For example the 2015 training datasets contain articles where
MeSH 2015 have been assigned. Also, for 2014, 2015 and 2016 there are two
versions (a and b) of the training data available. The small version (wrt size) consists
of articles that belong to the pool of journals that the BioASQ team used to
select the articles for the test data (this was a subset of the available journals).
The bigger version consists of articles from every available
journal. Since 2017 articles for the test data will be selected from all
available journals, so only one corresponding training data set will be
available. The evaluation of the results during each year of the challenge is
performed using the corresponding version of the MeSH terms, thus their usage
is highly recommended. The training datasets of previous years of the challenge
are also available for reference reasons. Note that not every MeSH term is
covered in the datasets.
""".format
_BIOASQ_2013A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2013, filename="allMeSH.zip")

_BIOASQ_2014A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2014, filename="allMeSH.zip")
_BIOASQ_2014bA_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2014, filename="allMeSH_limitjournals.zip")

_BIOASQ_2015A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2015, filename="allMeSH.zip")
_BIOASQ_2015bA_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2015, filename="allMeSH_limitjournals.zip")

_BIOASQ_2016A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2016, filename="allMeSH_2016.zip")
_BIOASQ_2016bA_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2016, filename="allMeSH_limitjournals_2016.zip")

_BIOASQ_2017A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2017, filename="allMeSH_2017.json")
_BIOASQ_2018A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2018, filename="allMeSH_2018.json")
_BIOASQ_2019A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2019, filename="allMeSH_2019.json")
_BIOASQ_2020A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2020, filename="allMeSH_2020.json")
_BIOASQ_2021A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2021, filename="allMeSH_2021.json")
_BIOASQ_2022A_DESCRIPTION = _DESCRIPTION_TEMPLATE(year=2022, filename="allMeSH_2022.json")

_DESCRIPTION = {
    "bioasq_2013a": _BIOASQ_2013A_DESCRIPTION,
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

_URLS = {
    "bioasq_2013a": "allMeSH.zip",
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
        "2013",
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
            "bioasq_2013a": "allMeSH.json",
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
            if self.config.subset_id in ("bioasq_2013a", "bioasq_2014a", "bioasq_2014ba", "bioasq_2015a", "bioasq_2015ba"):
                article_index = 0

                for line in f:
                    try:
                        record = json.loads(line.rstrip(",\n"))
                    except json.decoder.JSONDecodeError:
                        # NOTE: First and last line of 2013, 2014 do not contain valid JSON,
                        # but also not any relevant data (first line has a single quote
                        # and the term 'articles=[' and the last line contains
                        # closing brackets. We skip these irrelevant lines.
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
