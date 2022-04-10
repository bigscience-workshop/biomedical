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

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

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

_DESCRIPTION = """\
EHR-Rel is a novel open-source1 biomedical concept relatedness dataset consisting of 3630 concept pairs, six times more
than the largest existing dataset.  Instead of manually selecting and pairing concepts as done in previous work,
the dataset is sampled from EHRs to ensure concepts are relevant for the EHR concept retrieval task.
A detailed analysis of the concepts in the dataset reveals a far larger coverage compared to existing datasets.
"""

_HOMEPAGE = "https://aclanthology.org/2020.coling-main.577/"

_LICENSE = "Apache License 2.0"

_URLS = {
    _DATASETNAME: "https://github.com/babylonhealth/EHR-Rel/archive/refs/heads/master.zip",
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
            description="EHR-Rel source schema",
            schema="source",
            subset_id="ehr_rel",
        ),
        BigBioConfig(
            name="ehr_rel_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="EHR-Rel BigBio schema",
            schema="bigbio_pairs",
            subset_id="ehr_rel",
        ),
    ]

    DEFAULT_CONFIG_NAME = "ehr_rel_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "label": datasets.Value("string"),
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
        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls))
        data_dir = data_dir.joinpath("EHR-Rel-master")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir},
            ),
        ]

    def _generate_examples(self, data_dir: Path) -> Iterator[Tuple[str, Dict]]:
        uid = 1
        for file in data_dir.iterdir():
            # Ignore hidden files and annotation files - we just consider the brat text files
            if file.suffix == ".tsv":
                with open(file, encoding="utf-8") as csv_file:
                    csv_reader = csv.reader(csv_file, quotechar='"', delimiter="\t")
                    next(csv_reader, None)  # remove column headers
                    for id_, row in enumerate(csv_reader):
                        uid += 1
                        document_id = str(uid)  # .tsv files don't contain document_ids
                        text_1 = row[1]
                        text_2 = row[3]
                        label = row[9]

                        if self.config.schema == "source":
                            yield uid, {
                                "document_id": document_id,
                                "text_1": text_1,
                                "text_2": text_2,
                                "label": label,
                            }

                        elif self.config.schema == "bigbio_pairs":
                            yield uid, {
                                "id": uid,  # uid is an unique identifier for every record that starts from 1
                                "document_id": document_id,  # .tsv files don't contain document_ids
                                "text_1": text_1,
                                "text_2": text_2,
                                "label": label,
                            }
