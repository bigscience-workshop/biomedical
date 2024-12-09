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
PMC-Patients dataset consists of 4 tasks. One of the task is Patient Note Recognition (PNR).
PMC-Patients PNR dataset is modeled as a paragraph-level sequential labeling task,
similar to the named entity recognition (NER) task.
For each article, given input as a sequence of texts p1, p2, ..., pn, where n is the number of paragraphs,
the output is a sequence of BIO tags t1, t2, ..., tn.
"""

import json
import os
from typing import Dict, List, Tuple

import datasets

from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@misc{zhao2022pmcpatients,
      title={PMC-Patients: A Large-scale Dataset of Patient Notes and Relations Extracted from Case
          Reports in PubMed Central},
      author={Zhengyun Zhao and Qiao Jin and Sheng Yu},
      year={2022},
      eprint={2202.13876},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}"""

_DATASETNAME = "pmc_patients_pnr"
_DISPLAYNAME = "PMC-Patients Task 1: Patient Note Recognition (PNR)"

_DESCRIPTION = """\
PMC-Patients PNR is a paragraph-level sequential modeling to recognize patient notes.
"""

_HOMEPAGE = "https://github.com/zhao-zy15/PMC-Patients"

_LICENSE = Licenses.CC_BY_NC_SA_4p0

_URLS = {
    _DATASETNAME: "https://drive.google.com/u/0/uc?id=1vFCLy_CF8fxPDZvDtHPR6Dl6x9l0TyvW&export=download",
}

_SUPPORTED_TASKS = []

_SOURCE_VERSION = "1.2.0"

_BIGBIO_VERSION = "1.0.0"


class PMCPatientsPNRDataset(datasets.GeneratorBasedBuilder):
    """
    PMC-Patients dataset consists of 4 tasks. One of the task is Patient Note Recognition (PNR).
    PMC-Patients PNR dataset is modeled as a paragraph-level sequential labeling task,
    similar to the named entity recognition (NER) task.
    For each article, given input as a sequence of texts p1, p2, ..., pn, where n is the number of paragraphs,
    the output is a sequence of BIO tags t1, t2, ..., tn.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pmc_patients_pnr_source",
            version=SOURCE_VERSION,
            description="pmc_patients_pnr source schema",
            schema="source",
            subset_id="pmc_patients_pnr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pmc_patients_pnr_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "texts": [datasets.Value("string")],
                    "tags": [datasets.Value("string")],
                }
            )

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
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "datasets/task_1_patient_note_recognition/PNR_train.json",
                    ),
                    "split": "train",
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "datasets/task_1_patient_note_recognition/PNR_test.json",
                    ),
                    "split": "test",
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "datasets/task_1_patient_note_recognition/PNR_dev.json",
                    ),
                    "split": "dev",
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str, data_dir: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as j:
            file = json.load(j)

        if self.config.schema == "source":

            for uid, article in enumerate(file):
                feature_dict = {
                    "id": uid,
                    "texts": article["texts"],
                    "tags": article["tags"],
                }
                yield uid, feature_dict
