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
EHR-Rel is a novel open-source1 biomedical concept relatedness dataset consisting of 3630 concept pairs, six times more
 than the largest existing dataset.  Instead of manually selecting and pairing concepts as done in previous work,
 the dataset is sampled from EHRs to ensure concepts are relevant for the EHR concept retrieval task.
 A detailed analysis of the concepts in the dataset reveals a far larger coverage compared to existing datasets.
"""

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import datasets

from .bigbiohub import pairs_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = False
_CITATION = """\
@inproceedings{schulz-etal-2020-biomedical,
    title = {Biomedical Concept Relatedness {--} A large {EHR}-based benchmark},
    author = {Schulz, Claudia  and
      Levy-Kramer, Josh  and
      Van Assel, Camille  and
      Kepes, Miklos  and
      Hammerla, Nils},
    booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
    month = {dec},
    year = {2020},
    address = {Barcelona, Spain (Online)},
    publisher = {International Committee on Computational Linguistics},
    url = {https://aclanthology.org/2020.coling-main.577},
    doi = {10.18653/v1/2020.coling-main.577},
    pages = {6565--6575},
    }
"""

_DATASETNAME = "ehr_rel"
_DISPLAYNAME = "EHR-Rel"

_DESCRIPTION = """\
EHR-Rel is a novel open-source1 biomedical concept relatedness dataset consisting of 3630 concept pairs, six times more
than the largest existing dataset.  Instead of manually selecting and pairing concepts as done in previous work,
the dataset is sampled from EHRs to ensure concepts are relevant for the EHR concept retrieval task.
A detailed analysis of the concepts in the dataset reveals a far larger coverage compared to existing datasets.
"""

_HOMEPAGE = "https://github.com/babylonhealth/EHR-Rel"

_LICENSE = "Apache License 2.0"

_URLS = {
    _DATASETNAME: {
        "ehr_rel_a": "https://raw.githubusercontent.com/babylonhealth/EHR-Rel/master/EHR-RelA.tsv",
        "ehr_rel_b": "https://raw.githubusercontent.com/babylonhealth/EHR-Rel/master/EHR-RelB.tsv",
    },
}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]


_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class EHRRelDataset(datasets.GeneratorBasedBuilder):
    """Dataset for EHR-Rel Corpus"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="ehr_rel_source",
            version=SOURCE_VERSION,
            description="EHR-Rel combined source schema",
            schema="source",
            subset_id="ehr_rel",
        ),
        BigBioConfig(
            name="ehr_rel_a_source",
            version=SOURCE_VERSION,
            description="EHR-Rel-A source schema",
            schema="source",
            subset_id="ehr_rel_a",
        ),
        BigBioConfig(
            name="ehr_rel_b_source",
            version=SOURCE_VERSION,
            description="EHR-Rel-B source schema",
            schema="source",
            subset_id="ehr_rel_b",
        ),
        BigBioConfig(
            name="ehr_rel_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="EHR-Rel BigBio schema",
            schema="bigbio_pairs",
            subset_id="ehr_rel",
        ),
    ]

    DEFAULT_CONFIG_NAME = "ehr_rel_bigbio_pairs"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "snomed_id_1": datasets.Value("string"),
                    "snomed_label_1": datasets.Value("string"),
                    "snomed_id_2": datasets.Value("string"),
                    "snomed_label_2": datasets.Value("string"),
                    "rater_A": datasets.Value("string"),
                    "rater_B": datasets.Value("string"),
                    "rater_C": datasets.Value("string"),
                    "rater_D": datasets.Value("string"),
                    "rater_E": datasets.Value("string"),
                    "mean_rating": datasets.Value("string"),
                    "CUI_1": datasets.Value("string"),
                    "CUI_2": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_pairs":
            features = pairs_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        urls = (
            [urls[self.config.subset_id]]
            if self.config.subset_id in urls
            else list(urls.values())
        )
        paths = dl_manager.download(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"paths": paths},
            ),
        ]

    def _generate_examples(self, paths: List[str]) -> Iterator[Tuple[str, Dict]]:

        uid = -1  # want first instance to be 0

        for path in paths:
            document_id = Path(path).stem
            with open(path, encoding="utf-8", newline="") as csv_file:
                csv_reader = csv.reader(csv_file, quotechar='"', delimiter="\t")
                next(csv_reader, None)  # remove column headers
                for id_, row in enumerate(csv_reader):
                    uid += 1
                    (
                        snomed_id_1,
                        snomed_label_1,
                        snomed_id_2,
                        snomed_label_2,
                        rater_A,
                        rater_B,
                        rater_C,
                        rater_D,
                        rater_E,
                        mean_rating,
                        CUI_1,
                        CUI_2,
                    ) = row

                    if self.config.schema == "source":
                        yield uid, {
                            "document_id": document_id,
                            "snomed_id_1": snomed_id_1,
                            "snomed_label_1": snomed_label_1,
                            "snomed_id_2": snomed_id_1,
                            "snomed_label_2": snomed_label_2,
                            "rater_A": rater_A,
                            "rater_B": rater_B,
                            "rater_C": rater_C,
                            "rater_D": rater_D,
                            "rater_E": rater_E,
                            "mean_rating": mean_rating,
                            "CUI_1": CUI_1,
                            "CUI_2": CUI_2,
                        }

                    elif self.config.schema == "bigbio_pairs":
                        yield uid, {
                            "id": uid,
                            "document_id": document_id,
                            "text_1": snomed_label_1,
                            "text_2": snomed_label_2,
                            "label": mean_rating,
                        }
