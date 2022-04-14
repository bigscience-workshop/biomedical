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

from typing import Dict, List, Tuple

import datasets
import pandas as pd

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{10.1093/nar/gkaa952,
    author = {Chen, Qingyu and Allot, Alexis and Lu, Zhiyong},
    title = "{LitCovid: an open database of COVID-19 literature}",
    journal = {Nucleic Acids Research},
    volume = {49},
    number = {D1},
    pages = {D1534-D1540},
    year = {2020},
    month = {11},
    abstract = "{
        Since the outbreak of the current pandemic in 2020, there has been a rapid
        growth of published articles on COVID-19 and SARS-CoV-2, with about 10 000
        new articles added each month. This is causing an increasingly serious
        information overload, making it difficult for scientists, healthcare
        professionals and the general public to remain up to date on the latest
        SARS-CoV-2 and COVID-19 research. Hence, we developed LitCovid
        (https://www.ncbi.nlm.nih.gov/research/coronavirus/), a curated literature
        hub, to track up-to-date scientific information in PubMed. LitCovid is
        updated daily with newly identified relevant articles organized into curated
        categories. To support manual curation, advanced machine-learning and
        deep-learning algorithms have been developed, evaluated and integrated into
        the curation workflow. To the best of our knowledge, LitCovid is the
        first-of-its-kind COVID-19-specific literature resource, with all of its
        collected articles and curated data freely available. Since its release,
        LitCovid has been widely used, with millions of accesses by users worldwide
        for various information needs, such as evidence synthesis, drug discovery and
        text and data mining, among others.
    }",
    issn = {0305-1048},
    doi = {10.1093/nar/gkaa952},
    url = {https://doi.org/10.1093/nar/gkaa952},
    eprint = {https://academic.oup.com/nar/article-pdf/49/D1/D1534/35364553/gkaa952.pdf},
}
"""

_DATASETNAME = "bc7_litcovid"

_DESCRIPTION = """\
The training and development datasets contain the publicly-available
text of over 30 thousand COVID-19-related articles and their metadata
(e.g., title, abstract, journal). Articles in both datasets have been
manually reviewed and articles annotated by in-house models.
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/media/store/files/2021/TRACK5_pos_1_BC7_submission_221.pdf"

_LICENSE = "Unknown"

_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/BC7-LitCovid-"

_URLS = {
    _DATASETNAME: {
        "train": _BASE + "Train.csv",
        "validation": _BASE + "Dev.csv",
        "test": _BASE + "Test-GS.csv",
    },
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CLASS_NAMES = [
    "Epidemic Forecasting",
    "Treatment",
    "Prevention",
    "Mechanism",
    "Case Report",
    "Transmission",
    "Diagnosis",
]


class BC7LitCovidDataset(datasets.GeneratorBasedBuilder):
    """Track 5 - LitCovid track Multi-label topic classification for COVID-19 literature annotation"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bc7_litcovid_source",
            version=SOURCE_VERSION,
            description="bc7_litcovid source schema",
            schema="source",
            subset_id="bc7_litcovid",
        ),
        BigBioConfig(
            name="bc7_litcovid_bigbio_text",
            version=BIGBIO_VERSION,
            description="bc7_litcovid BigBio schema",
            schema="bigbio_text",
            subset_id="bc7_litcovid",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bc7_litcovid_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "journal": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "keywords": datasets.Sequence(datasets.Value("string")),
                    "pub_type": datasets.Sequence(datasets.Value("string")),
                    "authors": datasets.Sequence(datasets.Value("string")),
                    "doi": datasets.Value("string"),
                    "labels": datasets.Sequence(datasets.ClassLabel(names=_CLASS_NAMES)),
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

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        # Download all the CSV
        urls = _URLS[_DATASETNAME]
        path_train = dl_manager.download(urls["train"])
        path_validation = dl_manager.download(urls["validation"])
        path_test = dl_manager.download(urls["test"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path_train,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": path_validation,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": path_test,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        idx = 0

        # Load the CSV and convert it to the string format
        df = pd.read_csv(filepath, sep=",").astype(str).replace({"nan": None})

        for index, e in df.iterrows():

            if self.config.schema == "source":

                yield idx, {
                    "pmid": e["pmid"],
                    "journal": e["journal"],
                    "title": e["title"],
                    "abstract": e["abstract"],
                    "keywords": e["keywords"].split(";") if e["keywords"] is not None else [],
                    "pub_type": e["pub_type"].split(";") if e["pub_type"] is not None else [],
                    "authors": e["authors"].split(";") if e["authors"] is not None else [],
                    "doi": e["doi"],
                    "labels": e["label"].split(";"),
                }

            elif self.config.schema == "bigbio_text":

                yield idx, {
                    "id": idx,
                    "document_id": e["pmid"],
                    "text": e["abstract"],
                    "labels": e["label"].split(";"),
                }

            idx += 1
