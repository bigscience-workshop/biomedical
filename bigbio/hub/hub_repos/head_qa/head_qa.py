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

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import BigBioConfig, Tasks, qa_features

_LANGUAGES = ["English", "Spanish"]
_LICENSE = "MIT"
_LOCAL = False
_PUBMED = False

_CITATION = """\
@inproceedings{vilares-gomez-rodriguez-2019-head,
    title = "{HEAD}-{QA}: A Healthcare Dataset for Complex Reasoning",
    author = "Vilares, David  and G{\'o}mez-Rodr{\'i}guez, Carlos",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1092",
    doi = "10.18653/v1/P19-1092",
    pages = "960--966"
}
"""

_DATASETNAME = "head_qa"
_DISPLAYNAME = "HEAD-QA"

_DESCRIPTION = """\
HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to access a specialized position in the \
Spanish healthcare system, and are challenging even for highly specialized humans. They are designed by the \
Ministerio de Sanidad, Consumo y Bienestar Social.The dataset contains questions about following topics: medicine, \
nursing, psychology, chemistry, pharmacology and biology.
"""

_HOMEPAGE = "https://aghie.github.io/head-qa/"

_URLS = {
    "HEAD": "https://drive.usercontent.google.com/u/0/uc?id=1dUIqVwvoZAtbX_-z5axCoe97XNcFo1No&export=download",
    "HEAD_EN": "https://drive.usercontent.google.com/u/0/uc?id=1phryJg4FjCFkn0mSCqIOP2-FscAeKGV0&export=download",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class HeadQADataset(datasets.GeneratorBasedBuilder):
    """HEAD-QA: A Healthcare Dataset for Complex Reasoning"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="head_qa_en_source",
            version=SOURCE_VERSION,
            description="HeadQA English source schema",
            schema="source",
            subset_id="head_qa_en",
        ),
        BigBioConfig(
            name="head_qa_es_source",
            version=SOURCE_VERSION,
            description="HeadQA Spanish source schema",
            schema="source",
            subset_id="head_qa_es",
        ),
        BigBioConfig(
            name="head_qa_en_bigbio_qa",
            version=BIGBIO_VERSION,
            description="HeadQA English Question Answering BigBio schema",
            schema="bigbio_qa",
            subset_id="head_qa_en",
        ),
        BigBioConfig(
            name="head_qa_es_bigbio_qa",
            version=BIGBIO_VERSION,
            description="HeadQA Spanish Question Answering BigBio schema",
            schema="bigbio_qa",
            subset_id="head_qa_es",
        ),
    ]

    DEFAULT_CONFIG_NAME = "head_qa_en_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "year": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "qid": datasets.Value("int32"),
                    "qtext": datasets.Value("string"),
                    "ra": datasets.Value("int32"),
                    "answers": [
                        {
                            "aid": datasets.Value("int32"),
                            "atext": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_qa":
            features = qa_features
        else:
            raise NotImplementedError(f"Schema {self.config.schema} is not supported")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.subset_id == "head_qa_en":
            data_dir = Path(dl_manager.download_and_extract(_URLS["HEAD_EN"])) / "HEAD_EN"
            subset_name = "HEAD_EN"

        elif self.config.subset_id == "head_qa_es":
            data_dir = Path(dl_manager.download_and_extract(_URLS["HEAD"])) / "HEAD"
            subset_name = "HEAD"

        else:
            raise NotImplementedError(f"Subset {self.config.subset_id} is not supported")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "input_json_file": data_dir / f"train_{subset_name}.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "input_json_file": data_dir / f"dev_{subset_name}.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "input_json_file": data_dir / f"test_{subset_name}.json",
                },
            ),
        ]

    def _generate_examples(self, input_json_file: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            for key, example in self._generate_source_documents(input_json_file):
                yield key, example

        elif self.config.schema == "bigbio_qa":
            for key, example in self._generate_source_documents(input_json_file):
                yield self._source_to_qa(example)

    def _generate_source_documents(self, input_json_file: Path) -> Tuple[str, Dict]:
        """Generates source instances."""

        with input_json_file.open("r", encoding="utf8") as file_stream:
            head_qa = json.load(file_stream)

        for exam_id, exam in enumerate(head_qa["exams"]):
            content = head_qa["exams"][exam]
            name = content["name"].strip()
            year = content["year"].strip()
            category = content["category"].strip()

            for question in content["data"]:
                qid = int(question["qid"].strip())
                qtext = question["qtext"].strip()
                ra = int(question["ra"].strip())

                aids = [answer["aid"] for answer in question["answers"]]
                atexts = [answer["atext"].strip() for answer in question["answers"]]
                answers = [{"aid": aid, "atext": atext} for aid, atext in zip(aids, atexts)]

                instance_id = f"{exam_id}_{qid}"
                instance = {
                    "name": name,
                    "year": year,
                    "category": category,
                    "qid": qid,
                    "qtext": qtext,
                    "ra": ra,
                    "answers": answers,
                }

                yield instance_id, instance

    def _source_to_qa(self, example: Dict) -> Tuple[str, Dict]:
        """Converts a source example to BigBio example."""

        instance = {
            "id": example["name"] + "_qid_" + str(example["qid"]),
            "question_id": example["qid"],
            "document_id": None,
            "question": example["qtext"],
            "type": "multiple_choice",
            "choices": [answer["atext"] for answer in example["answers"]],
            "context": None,
            "answer": [next(filter(lambda answer: answer["aid"] == example["ra"], example["answers"]))["atext"]],
        }

        return instance["id"], instance
