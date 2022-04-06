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
import os
import json
import glob
import datasets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import utils.parsing as parsing
import utils.schemas as schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}
"""

_DATASETNAME = "pubmed_qa"

_DESCRIPTION = """\
PubMedQA is a novel biomedical question answering (QA) dataset collected from PubMed abstracts.
The task of PubMedQA is to answer research biomedical questions with yes/no/maybe using the corresponding abstracts. 
PubMedQA has 1k expert-annotated, 61.2k unlabeled and 211.3k artificially generated QA instances. 

Each PubMedQA instance is composed of:
  (1) a question which is either an existing research article title or derived from one, 
  (2) a context which is the corresponding abstract without its conclusion, 
  (3) a long answer, which is the conclusion of the abstract and, presumably, answers the research question, and
  (4) a yes/no/maybe answer which summarizes the conclusion.

PubMedQA is the first QA dataset where reasoning over biomedical research texts, 
especially their quantitative contents, is required to answer the questions.
"""

_HOMEPAGE = "https://github.com/pubmedqa/pubmedqa"
_LICENSE = "MIT License"

_URLS = {
    "pqaa": "https://drive.google.com/uc?export=download&id=1kaU0ECRbVkrfjBAKtVsPCRF6qXSouoq9",
    "pqal": "https://drive.google.com/uc?export=download&id=1kQnjowPHOcxETvYko7DRG9wE7217BQrD",
    "pqau": "https://drive.google.com/uc?export=download&id=1q4T_nhhj8UvJ9JbZedhkTZHN6ZeEZ2H9",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CLASS_NAMES = [
    "Yes",
    "No",
    "Maybe"
]

class PubmedQADataset(datasets.GeneratorBasedBuilder):
    """PubmedQA Dataset"""
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        # Source Schema
        BigBioConfig(
            name="pqal_source",
            version=SOURCE_VERSION,
            description="PubmedQA labeled source schema",
            schema="source",
            subset_id="pqal",
        ),
        BigBioConfig(
            name="pqaa_source",
            version=SOURCE_VERSION,
            description="PubmedQA artificial source schema",
            schema="source",
            subset_id="pqaa",
        ),
        BigBioConfig(
            name="pqau_source",
            version=SOURCE_VERSION,
            description="PubmedQA unlabeled source schema",
            schema="source",
            subset_id="pqau",
        ),
        # BigBio Schema
        BigBioConfig(
            name="pqal_bigbio_qa",
            version=BIGBIO_VERSION,
            description="PubmedQA labeled BigBio schema",
            schema="bigbio_qa",
            subset_id="pqal",
        ),
        BigBioConfig(
            name="pqaa_bigbio_qa",
            version=BIGBIO_VERSION,
            description="PubmedQA artificial BigBio schema",
            schema="bigbio_qa",
            subset_id="pqaa",
        ),
        BigBioConfig(
            name="pqau_bigbio_qa",
            version=BIGBIO_VERSION,
            description="PubmedQA unlabeled BigBio schema",
            schema="bigbio_qa",
            subset_id="pqau",
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "pqal_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "QUESTION": datasets.Value("string"),
                    "CONTEXTS": datasets.Sequence(datasets.Value("string")),
                    "LABELS": datasets.Sequence(datasets.Value("string")),
                    "MESHES": datasets.Sequence(datasets.Value("string")),
                    "YEAR": datasets.Value("string"),
                    "reasoning_required_pred": datasets.Value("string"),
                    "reasoning_free_pred": datasets.Value("string"),
                    "final_decision": datasets.Value("string"),
                    "LONG_ANSWER": datasets.Value("string")
                },
            )
        elif self.config.schema == "bigbio_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.subset_id]
        data_dir = Path(dl_manager.download_and_extract(urls))

        if self.config.subset_id == 'pqal':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": data_dir / "pqal_train_dev_set.json"
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": data_dir / "pqal_test_set.json"
                    }
                )
            ]            
        elif self.config.subset_id == 'pqaa':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": data_dir / "pqaa_train_set.json"
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": data_dir / "pqaa_dev_set.json"
                    }
                )
            ]            
        else: # if self.config.subset_id == 'pqau'
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": data_dir / "ori_pqau.json"
                    }
                )
            ]

    def _generate_examples(self, filepath: Path) -> Iterator[Tuple[str, Dict]]:
        data = json.load(open(filepath, 'r'))

        if self.config.schema == "source":
            for id, row in data.items():
                if self.config.subset_id == 'pqau':
                    row["reasoning_required_pred"] = None
                    row["reasoning_free_pred"] = None
                    row["final_decision"] = None
                elif self.config.subset_id == 'pqaa':
                    row["YEAR"] = None
                    row["reasoning_required_pred"] = None
                    row["reasoning_free_pred"] = None
                    
                yield id, row
        elif self.config.schema == "bigbio_qa":
            for id, row in data.items():
                if self.config.subset_id == 'pqau':
                    answers =  [row['LONG_ANSWER']]
                else:
                    answers = [row['final_decision'], row['LONG_ANSWER']]

                qa_row = {
                    "id": id,
                    "question_id": id,
                    "document_id": id,
                    "question": row['QUESTION'],
                    "type": 'abstractive',
                    "context": ' '.join(row['CONTEXTS']),
                    "answer": answers,
                }         
                
                yield id, qa_row
