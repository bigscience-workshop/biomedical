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
UMNSRS, developed by Pakhomov, et al., consists of 725 clinical term pairs whose semantic similarity and relatedness.
The similarity and relatedness of each term pair was annotated based on a continuous scale by having the resident touch
a bar on a touch sensitive computer screen to indicate the degree of similarity or relatedness. The Intraclass
Correlation Coefficient (ICC) for the reference standard tagged for similarity was 0.47, and 0.50 for relatedness.
Therefore, as suggested by Pakhomov and colleagues, the subset below consists of 401 pairs for the similarity set and
430 pairs for the relatedness set which each have an ICC equal to 0.73.
"""

import csv
from typing import Dict, List, Tuple

import datasets
from biomed_datasets.utils import schemas
from biomed_datasets.utils.configs import BigBioConfig
from biomed_datasets.utils.constants import Tasks

_CITATION = """\
@inproceedings{pakhomov2010semantic,
  title={Semantic similarity and relatedness between clinical terms: an experimental study},
  author={Pakhomov, Serguei and McInnes, Bridget and Adam, Terrence and Liu, Ying and Pedersen, Ted and Melton, \
  Genevieve B},
  booktitle={AMIA annual symposium proceedings},
  volume={2010},
  pages={572},
  year={2010},
  organization={American Medical Informatics Association}
}
"""

_DATASETNAME = "umnsrs"

_DESCRIPTION = """\
UMNSRS, developed by Pakhomov, et al., consists of 725 clinical term pairs whose semantic similarity and relatedness.
The similarity and relatedness of each term pair was annotated based on a continuous scale by having the resident touch
a bar on a touch sensitive computer screen to indicate the degree of similarity or relatedness. The Intraclass
Correlation Coefficient (ICC) for the reference standard tagged for similarity was 0.47, and 0.50 for relatedness.
Therefore, as suggested by Pakhomov and colleagues, the subset below consists of 401 pairs for the similarity set and
430 pairs for the relatedness set which each have an ICC equal to 0.73.
"""

_HOMEPAGE = "https://nlp.cs.vcu.edu/data.html#umnsrs"

_LICENSE = "Unknown"

_URLS = {
    "source": ["https://nlp.cs.vcu.edu/data/similarity-data/UMNSRS_similarity.csv"],
    "bigbio_pairs": ["https://nlp.cs.vcu.edu/data/similarity-data/UMNSRS_similarity.csv"],
}

_SUPPORTED_TASKS = [
    Tasks.SEMANTIC_SIMILARITY
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MayosrsDataset(datasets.GeneratorBasedBuilder):
    """UMNSRS, developed by Pakhomov, et al., consists of 725 clinical term pairs whose semantic similarity and
    relatedness."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="umnsrs_source",
            version=SOURCE_VERSION,
            description="UMNSRS source schema",
            schema="source",
            subset_id="umnsrs",
        ),
        BigBioConfig(
            name="umnsrs_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="UMNSRS BigBio schema",
            schema="bigbio_pairs",
            subset_id="umnsrs",
        ),
    ]

    DEFAULT_CONFIG_NAME = "umnsrs_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "mean_score": datasets.Value("float32"),
                    "std_score": datasets.Value("float32"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "code_1": datasets.Value("string"),
                    "code_2": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_pairs":
            features = schemas.pairs_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[self.config.schema]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if split == "train":
            combined_file = filepath[0]

            data = []
            with open(combined_file, encoding="utf-8") as csv_file:
                csv_reader = csv.reader(
                    csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
                )
                next(csv_reader)
                for id_, row in enumerate(csv_reader):
                    mean_score, std_score, text_1, text_2, code_1, code_2 = row
                    data.append([mean_score, std_score, text_1, text_2, code_1, code_2])

            if self.config.schema == "source":
                for id_, row in enumerate(data):
                    mean_score, std_score, text_1, text_2, code_1, code_2 = row

                    yield id_, {
                        "mean_score": float(mean_score),
                        "std_score": float(std_score),
                        "text_1": text_1,
                        "text_2": text_2,
                        "code_1": code_1,
                        "code_2": code_2,
                    }

            elif self.config.schema == "bigbio_pairs":
                uid = 0
                for id_, row in enumerate(data):
                    uid += 1
                    label, _, text_1, text_2, _, _ = row
                    yield id_, {
                        "id": uid,  # uid is an unique identifier for every record that starts from 1
                        "document_id": "NULL",
                        "text_1": text_1,
                        "text_2": text_2,
                        "label": str(label),
                    }

        else:
            print("There's no test/val split available for the given dataset")
            return
