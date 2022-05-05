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
Medical Question-Answer Summarization (MEDIQA-AnS): Question-Driven Summarization of Answers to Consumer Health
Questions. The first summarization collection containing question-driven summaries of answers to consumer health
questions. This dataset can be used to evaluate single or multi-document summaries generated by algorithms using
extractive or abstractive approaches.
This dataset contains 8 different subsets which support both source and bigbio schema. These subsets arise from
the permutation of three different settings:
- [page2answer, section2answer]: Full page as context / manually selected passages as context
- [multi, single]: Generate summary of answer for specific question across multiple documents / 
                   Generate summary of answer for specific question for each document (each document is sample)
- [abstractive, extractive]: Abstractive summary / extractive summary
List of subset_ids:
- mediqa_ans_page2answer_multi_abstractive
  This split contains the question ID and question, the answer ID and the full text of the web pages, the
  corresponding rating for each answer, and the multi-document abstractive summary. 
- mediqa_ans_page2answer_multi_extractive
  Contains the question ID and question, the answer ID and full text of the web pages, the corresponding rating
  for each answer, and the multi-document extractive summary. 
- mediqa_ans_page2answer_single_abstractive
  Contains the question ID and question, the answer ID and full text of the web pages, the corresponding rating
  for each answer, and the single document abstractive summary for each answer.
- mediqa_ans_page2answer_single_extractive
  Contains the question ID and question, the answer ID and full text of the web pages, the corresponding rating
  for each answer, and the single document extractive summary for each answer.
- mediqa_ans_section2answer_multi_abstractive
  Contains the question ID and question, the answer ID and manually selected passages, the corresponding rating
  for each answer, and the multi-document abstractive summary for each answer.
- mediqa_ans_section2answer_multi_extractive
  Contains the question ID and question, the answer ID and manually selected passages, the corresponding rating
  for each answer, and the multi-document extractive summary for each answer.
- mediqa_ans_section2answer_single_abstractive
  Contains the question ID and question, the answer ID and manually selected passages, the corresponding rating
  for each answer, and the single document abstractive summary for each answer.
- mediqa_ans_section2answer_single_extractive
  Contains the question ID and question, the answer ID and manually selected passages, the corresponding rating
  for each answer, and the single document extractive summary for each answer.
Furthermore there exists the subset mediqa_ans_all for which there only exists a source schema and contains
all questions, pages, passages, ratings, urls, and each type of summaries.
"""

import itertools as it
import json
import os
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_CITATION = """\
@article{,
    author={Savery, Max
        and Abacha, Asma Ben
        and Gayen, Soumya
        and Demner-Fushman, Dina},
    title={Question-driven summarization of answers to consumer health questions},
    journal={Scientific Data},
    year={2020},
    month={Oct},
    day={02},
    volume={7},
    number={1},
    pages={322},
    issn={2052-4463},
    doi={10.1038/s41597-020-00667-z},
    url={https://doi.org/10.1038/s41597-020-00667-z}
}
"""

_DATASETNAME = "mediqa_ans"

_DESCRIPTION = """\
Medical Question-Answer Summarization (MEDIQA-AnS): Question-Driven Summarization of Answers to Consumer Health
Questions. The first summarization collection containing question-driven summaries of answers to consumer health
questions. This dataset can be used to evaluate single or multi-document summaries generated by algorithms using
extractive or abstractive approaches.
This dataset contains 8 different subsets which support both source and bigbio schema. These subsets arise from
the permutation of three different settings:
- [page2answer, section2answer]: Full page as context / manually selected passages as context
- [multi, single]: Generate summary of answer for specific question across multiple documents / 
                   Generate summary of answer for specific question for each document (each document is sample)
- [abstractive, extractive]: Abstractive summary / extractive summary
List of subset_ids:
- mediqa_ans_page2answer_multi_abstractive
  This split contains the question ID and question, the answer ID and the full text of the web pages, the
  corresponding rating for each answer, and the multi-document abstractive summary. 
- mediqa_ans_page2answer_multi_extractive
  Contains the question ID and question, the answer ID and full text of the web pages, the corresponding rating
  for each answer, and the multi-document extractive summary. 
- mediqa_ans_page2answer_single_abstractive
  Contains the question ID and question, the answer ID and full text of the web pages, the corresponding rating
  for each answer, and the single document abstractive summary for each answer.
- mediqa_ans_page2answer_single_extractive
  Contains the question ID and question, the answer ID and full text of the web pages, the corresponding rating
  for each answer, and the single document extractive summary for each answer.
- mediqa_ans_section2answer_multi_abstractive
  Contains the question ID and question, the answer ID and manually selected passages, the corresponding rating
  for each answer, and the multi-document abstractive summary for each answer.
- mediqa_ans_section2answer_multi_extractive
  Contains the question ID and question, the answer ID and manually selected passages, the corresponding rating
  for each answer, and the multi-document extractive summary for each answer.
- mediqa_ans_section2answer_single_abstractive
  Contains the question ID and question, the answer ID and manually selected passages, the corresponding rating
  for each answer, and the single document abstractive summary for each answer.
- mediqa_ans_section2answer_single_extractive
  Contains the question ID and question, the answer ID and manually selected passages, the corresponding rating
  for each answer, and the single document extractive summary for each answer.
Furthermore there exists the subset mediqa_ans_all for which there only exists a source schema and contains
all questions, pages, passages, ratings, urls, and each type of summaries.
"""

_HOMEPAGE = "https://osf.io/fyg46/"

_LICENSE = "CC0"

_URLS = {
    _DATASETNAME: "https://osf.io/fs57e/download",
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MediqaAnsDataset(datasets.GeneratorBasedBuilder):
    """
    A dataset of manually generated, question-driven summaries of multi and
    single document answers to consumer health questions.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    BUILDER_CONFIGS.append(
        BigBioConfig(
            name="mediqa_ans_all_source",
            version=BIGBIO_VERSION,
            description="MEDIQA-AnS All source schema",
            schema="source",
            subset_id="mediqa_ans_all",
        ),
    )

    for setting1 in ["page2answer", "section2answer"]:
        for setting2 in ["multi", "single"]:
            for setting3 in ["abstractive", "extractive"]:
                BUILDER_CONFIGS.append(
                    BigBioConfig(
                        name=f"mediqa_ans_{setting1}_{setting2}_{setting3}_bigbio_t2t",
                        version=BIGBIO_VERSION,
                        description=f"MEDIQA-AnS {setting1} {setting2} {setting3} BigBio schema",
                        schema="bigbio_t2t",
                        subset_id=f"mediqa_ans_{setting1}_{setting2}_{setting3}",
                    )
                )
                BUILDER_CONFIGS.append(
                    BigBioConfig(
                        name=f"mediqa_ans_{setting1}_{setting2}_{setting3}_source",
                        version=BIGBIO_VERSION,
                        description=f"MEDIQA-AnS {setting1} {setting2} {setting3} source schema",
                        schema="source",
                        subset_id=f"mediqa_ans_{setting1}_{setting2}_{setting3}",
                    ),
                )

    DEFAULT_CONFIG_NAME = "mediqa_ans_page2answer_multi_abstractive_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source" and self.config.subset_id == "mediqa_ans_all":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "multi_abs_summ": datasets.Value("string"),
                    "multi_ext_summ": datasets.Value("string"),
                    "answers": [
                        {
                            "id": datasets.Value("string"),
                            "answer_abs_summ": datasets.Value("string"),
                            "answer_ext_summ": datasets.Value("string"),
                            "section": datasets.Value("string"),
                            "article": datasets.Value("string"),
                            "url": datasets.Value("string"),
                            "rating": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "source":
            features = datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                    "articles": [
                        {
                            "answer_id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "rating": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_t2t":
            features = schemas.text2text_features

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
        file_path = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(file_path),
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        dataset = None
        with open(filepath, "r", encoding="utf8") as infile:
            dataset = json.load(infile)

        uid = it.count(0)
        if self.config.name == "mediqa_ans_all_source":
            dataset = self._json_dict_to_list(dataset, "id")
            for example in dataset:
                example["answers"] = self._json_dict_to_list(example["answers"], "id")
                yield example["id"], example
        else:
            _, setting1, setting2, setting3 = self.config.subset_id.rsplit("_", 3)
            if self.config.schema == "source":
                for example in self._generate_setting_examples(dataset, setting1, setting2, setting3):
                    yield next(uid), example
            elif self.config.schema == "bigbio_t2t":
                for example in self._generate_setting_examples(dataset, setting1, setting2, setting3):
                    example = self._source_to_t2t(example)
                    example["id"] = next(uid)
                    yield example["id"], example

    def _generate_setting_examples(self, dataset, setting1, setting2, setting3):
        for question_id, question in dataset.items():
            example = {}
            example["question_id"] = question_id
            example["question"] = question["question"]
            if setting2 == "single":
                for answer_id, answer in question["answers"].items():
                    example_ = example.copy()
                    if setting1 == "section2answer":
                        example_["articles"] = [
                            {"answer_id": answer_id, "text": answer["section"], "rating": answer["rating"]}
                        ]
                    elif setting1 == "page2answer":
                        example_["articles"] = [
                            {"answer_id": answer_id, "text": answer["article"], "rating": answer["rating"]}
                        ]
                    if setting3 == "abstractive":
                        example_["summary"] = answer["answer_abs_summ"]
                    elif setting3 == "extractive":
                        example_["summary"] = answer["answer_ext_summ"]
                    yield example_
            elif setting2 == "multi":
                example["articles"] = []
                for answer_id, answer in question["answers"].items():
                    if setting1 == "section2answer":
                        example["articles"].append(
                            {"answer_id": answer_id, "text": answer["section"], "rating": answer["rating"]}
                        )
                    elif setting1 == "page2answer":
                        example["articles"].append(
                            {"answer_id": answer_id, "text": answer["article"], "rating": answer["rating"]}
                        )

                if setting3 == "abstractive":
                    example["summary"] = question["multi_abs_summ"]
                elif setting3 == "extractive":
                    example["summary"] = question["multi_ext_summ"]
                yield example

    def _source_to_t2t(self, example):
        example_ = {}
        example_["document_id"] = ""
        example_["text_1_name"] = ""
        example_["text_2_name"] = ""

        text1 = ""
        text1 += "Question ID: " + example["question_id"] + "\n"
        text1 += "Question: " + example["question"] + "\n"
        for article in example["articles"]:
            text1 += "Answer ID: " + article["answer_id"] + "\n"
            text1 += "Answer: " + article["text"] + "\n"
            text1 += "Rating: " + article["rating"] + "\n"
        example_["text_1"] = text1

        example_["text_2"] = example["summary"]

        return example_

    def _json_dict_to_list(self, json, new_key):
        list_ = []
        for key, values in json.items():
            assert isinstance(values, dict), "Child element is not a dict"
            assert new_key not in values, "New key already in values"
            values[new_key] = key
            list_.append(values)
        return list_