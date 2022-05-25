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
In this work, we present the first free-form multiple-choice OpenQA dataset for solving medical problems, MedQA,
collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and
traditional Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively. Together
with the question data, we also collect and release a large-scale corpus from medical textbooks from which the reading
comprehension models can obtain necessary knowledge for answering the questions.
"""

import os
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

# TODO: Add BibTeX citation
_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@article{,
    AUTHOR = {Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
    TITLE = {What Disease Does This Patient Have? A Large-Scale Open Domain Question \
             Answering Dataset from Medical Exams},
    JOURNAL = {Applied Sciences},
    VOLUME = {11},
    YEAR = {2021},
    NUMBER = {14},
    ARTICLE-NUMBER = {6421},
    URL = {https://www.mdpi.com/2076-3417/11/14/6421},
    ISSN = {2076-3417},
    DOI = {10.3390/app11146421}
}
"""

_DATASETNAME = "med_qa"

_DESCRIPTION = """\
In this work, we present the first free-form multiple-choice OpenQA dataset for solving medical problems, MedQA,
collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and
traditional Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively. Together
with the question data, we also collect and release a large-scale corpus from medical textbooks from which the reading
comprehension models can obtain necessary knowledge for answering the questions.
"""

_HOMEPAGE = "https://github.com/jind11/MedQA"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: "https://drive.google.com/u/0/uc?export=download&confirm=t&id=1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

_SUBSET2NAME = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "tw": "Chinese (Traditional, Taiwan)",
    "tw_en": "Chinese (Traditional, Taiwan) translated to English",
    "tw_zh": "Chinese (Traditional, Taiwan) translated to Chinese (Simplified)",
}


class MedQADataset(datasets.GeneratorBasedBuilder):
    """Free-form multiple-choice OpenQA dataset covering three languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    for subset in ["en", "zh", "tw", "tw_en", "tw_zh"]:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"med_qa_{subset}_source",
                version=SOURCE_VERSION,
                description=f"MedQA {_SUBSET2NAME.get(subset)} source schema",
                schema="source",
                subset_id=f"med_qa_{subset}",
            )
        )
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"med_qa_{subset}_bigbio_qa",
                version=BIGBIO_VERSION,
                description=f"MedQA {_SUBSET2NAME.get(subset)} BigBio schema",
                schema="bigbio_qa",
                subset_id=f"med_qa_{subset}",
            )
        )

    DEFAULT_CONFIG_NAME = "med_qa_en_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "meta_info": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer_idx": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "options": [
                        {
                            "key": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                }
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

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        lang_dict = {"en": "US", "zh": "Mainland", "tw": "Taiwan"}
        base_dir = os.path.join(data_dir, "data_clean", "questions")
        if self.config.subset_id in ["med_qa_en", "med_qa_zh", "med_qa_tw"]:
            lang_path = lang_dict.get(self.config.subset_id.rsplit("_", 1)[1])
            paths = {
                "train": os.path.join(base_dir, lang_path, "train.jsonl"),
                "test": os.path.join(base_dir, lang_path, "test.jsonl"),
                "valid": os.path.join(base_dir, lang_path, "dev.jsonl"),
            }
        elif self.config.subset_id == "med_qa_tw_en":
            paths = {
                "train": os.path.join(base_dir, "Taiwan", "tw_translated_jsonl", "en", "train-2en.jsonl"),
                "test": os.path.join(base_dir, "Taiwan", "tw_translated_jsonl", "en", "test-2en.jsonl"),
                "valid": os.path.join(base_dir, "Taiwan", "tw_translated_jsonl", "en", "dev-2en.jsonl"),
            }
        elif self.config.subset_id == "med_qa_tw_zh":
            paths = {
                "train": os.path.join(base_dir, "Taiwan", "tw_translated_jsonl", "zh", "train-2zh.jsonl"),
                "test": os.path.join(base_dir, "Taiwan", "tw_translated_jsonl", "zh", "test-2zh.jsonl"),
                "valid": os.path.join(base_dir, "Taiwan", "tw_translated_jsonl", "zh", "dev-2zh.jsonl"),
            }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": paths["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": paths["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": paths["valid"],
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        print(filepath)
        data = pd.read_json(filepath, lines=True)

        if self.config.schema == "source":
            for key, example in data.iterrows():
                example = example.to_dict()
                example["options"] = [{"key": key, "value": value} for key, value in example["options"].items()]
                yield key, example

        elif self.config.schema == "bigbio_qa":
            for key, example in data.iterrows():
                example = example.to_dict()
                example_ = {}
                example_["id"] = key
                example_["question_id"] = key
                example_["document_id"] = key
                example_["question"] = example["question"]
                example_["type"] = "multiple_choice"
                example_["choices"] = [value for value in example["options"].values()]
                example_["context"] = ""
                example_["answer"] = [example["answer"]]
                yield key, example_
