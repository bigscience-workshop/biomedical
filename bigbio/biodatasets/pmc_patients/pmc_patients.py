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
PPS dataset is a list of triplets. Each entry is in format (patient_uid_1, patient_uid_2, similarity)
where similarity has three values:0, 1, 2, indicating corresponding similarity.
"""

import json
import os
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = []
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

_DATASETNAME = "pmc_patients"

_DESCRIPTION = """\
This dataset is used for calculating the similarity between two patient descriptions.
"""

_HOMEPAGE = "https://github.com/zhao-zy15/PMC-Patients"

_LICENSE = Licenses.CC_BY_NC_SA_4p0

_URLS = {
    _DATASETNAME: "https://drive.google.com/u/0/uc?id=1vFCLy_CF8fxPDZvDtHPR6Dl6x9l0TyvW&export=download",
}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]

_SOURCE_VERSION = "1.2.0"

_BIGBIO_VERSION = "1.0.0"


class PMCPatientsDataset(datasets.GeneratorBasedBuilder):
    """PPS dataset is a list of triplets.
    Each entry is in format (patient_uid_1, patient_uid_2, similarity) and their
    respective texts.
    where similarity has three values:0, 1, 2, indicating corresponding similarity.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pmc_patients_source",
            version=SOURCE_VERSION,
            description="pmc_patients source schema",
            schema="source",
            subset_id="pmc_patients",
        ),
        BigBioConfig(
            name="pmc_patients_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="pmc_patients BigBio schema",
            schema="bigbio_pairs",
            subset_id="pmc_patients",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pmc_patients_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "id_text1": datasets.Value("string"),
                    "id_text2": datasets.Value("string"),
                    "label": datasets.Value("int8"),
                }
            )

        elif self.config.schema == "bigbio_pairs":
            features = schemas.pairs_features

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
                        "datasets/task_2_patient2patient_similarity/PPS_train.json",
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
                        "datasets/task_2_patient2patient_similarity/PPS_test.json",
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
                        "datasets/task_2_patient2patient_similarity/PPS_dev.json",
                    ),
                    "split": "dev",
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(
        self, filepath, split: str, data_dir: str
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        uid = 0

        def lookup_text(patient_uid: str, df: pd.DataFrame) -> str:
            try:
                return df.loc[patient_uid]["patient"]
            except KeyError:
                return ""

        with open(filepath, "r") as j:
            ret_file = json.load(j)

        if self.config.schema == "source":

            for key, (id1, id2, label) in enumerate(ret_file):
                feature_dict = {
                    "id": uid,
                    "id_text1": id1,
                    "id_text2": id2,
                    "label": label,
                }
                uid += 1
                yield key, feature_dict

        elif self.config.schema == "bigbio_pairs":
            source_files = os.path.join(data_dir, f"datasets/PMC-Patients_{split}.json")
            src_frame = pd.read_json(source_files, encoding="utf8").set_index(
                "patient_uid"
            )
            for key, (id1, id2, label) in enumerate(ret_file):
                text_1 = lookup_text(id1, src_frame)
                text_2 = lookup_text(id2, src_frame)
                # test/dev splits are faulty and may not contain the patient_uid
                # if any of the lookup texts are empty skip the sample
                if text_1 == "" or text_2 == "":
                    continue
                feature_dict = {
                    "id": uid,
                    "document_id": "NULL",
                    "text_1": text_1,
                    "text_2": text_2,
                    "label": label,
                }
                uid += 1
                yield key, feature_dict
