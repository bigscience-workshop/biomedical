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
Natural Language Inference (NLI) is the task of determining whether a given hypothesis can be
inferred from a given premise. Also known as Recognizing Textual Entailment (RTE), this task has
enjoyed popularity among researchers for some time. However, almost all datasets for this task
focused on open domain data such as as news texts, blogs, and so on. To address this gap, the MedNLI
dataset was created for language inference in the medical domain. MedNLI is a derived dataset with
data sourced from MIMIC-III v1.4. In order to stimulate research for this problem, a shared task on
Medical Inference and Question Answering (MEDIQA) was organized at the workshop for biomedical
natural language processing (BioNLP) 2019. The dataset provided herein is a test set of 405 premise
hypothesis pairs for the NLI challenge in the MEDIQA shared task. Participants of the shared task
are expected to use the MedNLI data for development of their models and this dataset was used as an
unseen dataset for scoring each participant submission.

The files comprising this dataset must be on the users local machine in a single directory that is
passed to `datasets.load_datset` via the `data_dir` kwarg. This loader script will read the archive
files directly (i.e. the user should not uncompress, untar or unzip any of the files). For example,
if `data_dir` is `"mediqa_nli"` it should contain the following files:

mediqa_nli
├── mednli-for-shared-task-at-acl-bionlp-2019-1.0.1.zip
"""

import json
import os
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = True
_CITATION = """\
@misc{https://doi.org/10.13026/gtv4-g455,
    title        = {MedNLI for Shared Task at ACL BioNLP 2019},
    author       = {Shivade,  Chaitanya},
    year         = 2019,
    publisher    = {physionet.org},
    doi          = {10.13026/GTV4-G455},
    url          = {https://physionet.org/content/mednli-bionlp19/}
}

"""

_DATASETNAME = "mediqa_nli"

_DESCRIPTION = """\
Natural Language Inference (NLI) is the task of determining whether a given hypothesis can be
inferred from a given premise. Also known as Recognizing Textual Entailment (RTE), this task has
enjoyed popularity among researchers for some time. However, almost all datasets for this task
focused on open domain data such as as news texts, blogs, and so on. To address this gap, the MedNLI
dataset was created for language inference in the medical domain. MedNLI is a derived dataset with
data sourced from MIMIC-III v1.4. In order to stimulate research for this problem, a shared task on
Medical Inference and Question Answering (MEDIQA) was organized at the workshop for biomedical
natural language processing (BioNLP) 2019. The dataset provided herein is a test set of 405 premise
hypothesis pairs for the NLI challenge in the MEDIQA shared task. Participants of the shared task
are expected to use the MedNLI data for development of their models and this dataset was used as an
unseen dataset for scoring each participant submission.
"""


_HOMEPAGE = "https://physionet.org/content/mednli-bionlp19/1.0.1/"
_LICENSE = "PhysioNet Credentialed Health Data License 1.5.0"

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.1"
_BIGBIO_VERSION = "1.0.0"


class MEDIQANLIDataset(datasets.GeneratorBasedBuilder):
    """MEDIQA NLI"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="mediqa_nli_source",
            version=SOURCE_VERSION,
            description="MEDIQA NLI source schema",
            schema="source",
            subset_id="mediqa_nli",
        ),
        BigBioConfig(
            name="mediqa_nli_bigbio_te",
            version=BIGBIO_VERSION,
            description="MEDIQA NLI BigBio schema",
            schema="bigbio_te",
            subset_id="mediqa_nli",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mediqa_nli_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pairID": datasets.Value("string"),
                    "gold_label": datasets.Value("string"),
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                    "sentence1_parse": datasets.Value("string"),
                    "sentence2_parse": datasets.Value("string"),
                    "sentence1_binary_parse": datasets.Value("string"),
                    "sentence2_binary_parse": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_te":
            features = schemas.entailment_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            extract_dir = dl_manager.extract(
                os.path.join(
                    self.config.data_dir,
                    "mednli-for-shared-task-at-acl-bionlp-2019-1.0.1.zip",
                )
            )
            data_dir = os.path.join(
                extract_dir, "mednli-for-shared-task-at-acl-bionlp-2019-1.0.1"
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "examples_filepath": os.path.join(
                        data_dir, "mednli_bionlp19_shared_task.jsonl"
                    ),
                    "ground_truth_filepath": os.path.join(
                        data_dir, "mednli_bionlp19_shared_task_ground_truth.csv"
                    ),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(
        self, examples_filepath: str, ground_truth_filepath: str, split: str
    ) -> Tuple[int, Dict]:

        ground_truth = pd.read_csv(
            ground_truth_filepath, index_col=0, squeeze=True
        ).to_dict()
        with open(examples_filepath, "r") as f:
            if self.config.schema == "source":
                for line in f:
                    json_line = json.loads(line)
                    json_line["gold_label"] = ground_truth[json_line["pairID"]]
                    yield json_line["pairID"], json_line

            elif self.config.schema == "bigbio_te":
                for line in f:
                    json_line = json.loads(line)
                    entailment_example = {
                        "id": json_line["pairID"],
                        "premise": json_line["sentence1"],
                        "hypothesis": json_line["sentence2"],
                        "label": ground_truth[json_line["pairID"]],
                    }
                    yield json_line["pairID"], entailment_example
