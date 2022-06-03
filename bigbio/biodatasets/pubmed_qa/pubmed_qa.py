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

# TODO: see if we can add long answer for QA task and text classification for MESH tags

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import datasets

import bigbio.utils.parsing as parsing
import bigbio.utils.schemas as schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import BigBioValues, Lang, Tasks

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
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
PubMedQA has 1k expert-annotated (PQA-L), 61.2k unlabeled (PQA-U) and 211.3k artificially generated QA instances (PQA-A).

Each PubMedQA instance is composed of:
  (1) a question which is either an existing research article title or derived from one,
  (2) a context which is the corresponding PubMed abstract without its conclusion,
  (3) a long answer, which is the conclusion of the abstract and, presumably, answers the research question, and
  (4) a yes/no/maybe answer which summarizes the conclusion.

PubMedQA is the first QA dataset where reasoning over biomedical research texts,
especially their quantitative contents, is required to answer the questions.

PubMedQA datasets comprise of 3 different subsets:
  (1) PubMedQA Labeled (PQA-L): A labeled PubMedQA subset comprises of 1k manually annotated yes/no/maybe QA data collected from PubMed articles.
  (2) PubMedQA Artificial (PQA-A): An artificially labelled PubMedQA subset comprises of 211.3k PubMed articles with automatically generated questions from the statement titles and yes/no answer labels generated using a simple heuristic.
  (3) PubMedQA Unlabeled (PQA-U): An unlabeled PubMedQA subset comprises of 61.2k context-question pairs data collected from PubMed articles.
"""

_HOMEPAGE = "https://github.com/pubmedqa/pubmedqa"
_LICENSE = Licenses.MIT
_URLS = {
    "pubmed_qa_artificial": "https://drive.google.com/uc?export=download&id=1kaU0ECRbVkrfjBAKtVsPCRF6qXSouoq9",
    "pubmed_qa_labeled": "https://drive.google.com/uc?export=download&id=1kQnjowPHOcxETvYko7DRG9wE7217BQrD",
    "pubmed_qa_unlabeled": "https://drive.google.com/uc?export=download&id=1q4T_nhhj8UvJ9JbZedhkTZHN6ZeEZ2H9",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CLASS_NAMES = ["yes", "no", "maybe"]


class PubmedQADataset(datasets.GeneratorBasedBuilder):
    """PubmedQA Dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = (
        [
            # PQA-A Source
            BigBioConfig(
                name="pubmed_qa_artificial_source",
                version=SOURCE_VERSION,
                description="PubmedQA artificial source schema",
                schema="source",
                subset_id="pubmed_qa_artificial",
            ),
            # PQA-U Source
            BigBioConfig(
                name="pubmed_qa_unlabeled_source",
                version=SOURCE_VERSION,
                description="PubmedQA unlabeled source schema",
                schema="source",
                subset_id="pubmed_qa_unlabeled",
            ),
            # PQA-A BigBio Schema
            BigBioConfig(
                name="pubmed_qa_artificial_bigbio_qa",
                version=BIGBIO_VERSION,
                description="PubmedQA artificial BigBio schema",
                schema="bigbio_qa",
                subset_id="pubmed_qa_artificial",
            ),
            # PQA-U BigBio Schema
            BigBioConfig(
                name="pubmed_qa_unlabeled_bigbio_qa",
                version=BIGBIO_VERSION,
                description="PubmedQA unlabeled BigBio schema",
                schema="bigbio_qa",
                subset_id="pubmed_qa_unlabeled",
            ),
        ]
        + [
            # PQA-L Source Schema
            BigBioConfig(
                name=f"pubmed_qa_labeled_fold{i}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description="PubmedQA labeled source schema",
                schema="source",
                subset_id=f"pubmed_qa_labeled_fold{i}",
            )
            for i in range(10)
        ]
        + [
            # PQA-L BigBio Schema
            BigBioConfig(
                name=f"pubmed_qa_labeled_fold{i}_bigbio_qa",
                version=datasets.Version(_BIGBIO_VERSION),
                description="PubmedQA labeled BigBio schema",
                schema="bigbio_qa",
                subset_id=f"pubmed_qa_labeled_fold{i}",
            )
            for i in range(10)
        ]
    )

    DEFAULT_CONFIG_NAME = "pubmed_qa_artificial_source"

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
                    "LONG_ANSWER": datasets.Value("string"),
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
        url_id = self.config.subset_id
        if "pubmed_qa_labeled" in url_id:
            # Enforce naming since there is fold number in the PQA-L subset
            url_id = "pubmed_qa_labeled"

        urls = _URLS[url_id]
        data_dir = Path(dl_manager.download_and_extract(urls))

        if "pubmed_qa_labeled" in self.config.subset_id:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": data_dir
                        / self.config.subset_id.replace("pubmed_qa_labeled", "pqal")
                        / "train_set.json"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": data_dir
                        / self.config.subset_id.replace("pubmed_qa_labeled", "pqal")
                        / "dev_set.json"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": data_dir / "pqal_test_set.json"},
                ),
            ]
        elif self.config.subset_id == "pubmed_qa_artificial":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": data_dir / "pqaa_train_set.json"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": data_dir / "pqaa_dev_set.json"},
                ),
            ]
        else:  # if self.config.subset_id == 'pubmed_qa_unlabeled'
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": data_dir / "ori_pqau.json"},
                )
            ]

    def _generate_examples(self, filepath: Path) -> Iterator[Tuple[str, Dict]]:
        data = json.load(open(filepath, "r"))

        if self.config.schema == "source":
            for id, row in data.items():
                if self.config.subset_id == "pubmed_qa_unlabeled":
                    row["reasoning_required_pred"] = None
                    row["reasoning_free_pred"] = None
                    row["final_decision"] = None
                elif self.config.subset_id == "pubmed_qa_artificial":
                    row["YEAR"] = None
                    row["reasoning_required_pred"] = None
                    row["reasoning_free_pred"] = None

                yield id, row
        elif self.config.schema == "bigbio_qa":
            for id, row in data.items():
                if self.config.subset_id == "pubmed_qa_unlabeled":
                    answers = [BigBioValues.NULL]
                else:
                    answers = [row["final_decision"]]

                qa_row = {
                    "id": id,
                    "question_id": id,
                    "document_id": id,
                    "question": row["QUESTION"],
                    "type": "yesno",
                    "choices": ["yes", "no", "maybe"],
                    "context": " ".join(row["CONTEXTS"]),
                    "answer": answers,
                }

                yield id, qa_row
