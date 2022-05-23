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
import xml.etree.ElementTree as ET

import bigbio.utils.parsing as parsing
import bigbio.utils.schemas as schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@inproceedings{MEDIQA2019,
  author    = {Asma {Ben Abacha} and Chaitanya Shivade and Dina Demner{-}Fushman},
  title     = {Overview of the MEDIQA 2019 Shared Task on Textual Inference, Question Entailment and Question Answering},
  booktitle = {ACL-BioNLP 2019},
  year      = {2019}
}
"""

_DATASETNAME = "mediqa_qa"

_DESCRIPTION = """\
The MEDIQA challenge is an ACL-BioNLP 2019 shared task aiming to attract further research efforts in Natural Language Inference (NLI), Recognizing Question Entailment (RQE), and their applications in medical Question Answering (QA).
Mailing List: https://groups.google.com/forum/#!forum/bionlp-mediqa

In the QA task, participants are tasked to:
- filter/classify the provided answers (1: correct, 0: incorrect).
- re-rank the answers.
"""

_HOMEPAGE = "https://sites.google.com/view/mediqa2019"
_LICENSE = "-"

_URLS = {
    _DATASETNAME: "https://github.com/abachaa/MEDIQA2019/archive/refs/heads/master.zip"
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

class PubmedQADataset(datasets.GeneratorBasedBuilder):
    """PubmedQA Dataset"""
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        # Source Schema
        BigBioConfig(
            name="mediqa_qa_source",
            version=SOURCE_VERSION,
            description="MEDIQA QA labeled source schema",
            schema="source",
            subset_id="mediqa_qa_source",
        ),
        # BigBio Schema
        BigBioConfig(
            name="mediqa_qa_bigbio_qa",
            version=BIGBIO_VERSION,
            description="MEDIQA QA BigBio schema",
            schema="bigbio_qa",
            subset_id="mediqa_qa_bigbio_qa",
        )
    ]

    DEFAULT_CONFIG_NAME = "mediqa_qa_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "QUESTION": {
                        "QID": datasets.Value("string"),
                        "QuestionText": datasets.Value("string"),
                        "AnswerList": [
                            {
                                "Answer": {
                                    "AID": datasets.Value("string"),
                                    "SystemRank": datasets.Value("int32"),
                                    "ReferenceRank": datasets.Value("int32"),
                                    "ReferenceScore": datasets.Value("int32"),
                                    "AnswerURL": datasets.Value("string"),
                                    "AnswerText": datasets.Value("string")
                                }
                            }
                        ]
                    }
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

    def _split_generators(self, dl_manager):
        data_dir = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME]))

        return [
            datasets.SplitGenerator(
                name=f'train_live_qa_med',
                gen_kwargs={
                    "filepath": data_dir / "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TrainingSet1-LiveQAMed.xml"
                }
            ),
            datasets.SplitGenerator(
                name=f'train_alexa',
                gen_kwargs={
                    "filepath": data_dir / "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TrainingSet2-Alexa.xml"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir / "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-ValidationSet.xml"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir / "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TestSet-wLabels.xml"
                }
            )
        ]

    def _generate_examples(self, filepath: Path) -> Iterator[Tuple[str, Dict]]:
        dom = ET.parse(filepath).getroot()
        for row in dom.iterfind('Question'):
            qid = row.attrib['QID']
            question_text = row.find('QuestionText').text

            answer_list = row.find('AnswerList')
            aids, sys_ranks, ref_ranks, ref_scores, answer_urls, answer_texts = [], [], [], [], [], []
            for answer in answer_list.findall('Answer'):
                aids.append(answer.attrib['AID'])
                sys_ranks.append(int(answer.attrib['SystemRank']))
                ref_ranks.append(int(answer.attrib['ReferenceRank']))
                ref_scores.append(int(answer.attrib['ReferenceScore']))
                answer_urls.append(answer.find('AnswerURL').text)
                answer_texts.append(answer.find('AnswerText').text)

            if self.config.schema == "source":
                yield qid, {
                    "QUESTION": {
                        "QID": qid,
                        "QuestionText": question_text,
                        "AnswerList": [
                            {
                                "Answer": {
                                    "AID": aid,
                                    "SystemRank": sys_rank,
                                    "ReferenceRank": ref_rank,
                                    "ReferenceScore": ref_score,
                                    "AnswerURL": ans_url,
                                    "AnswerText": ans_text
                                }
                            } for (aid, sys_rank, ref_rank, ref_score, ans_url, ans_text) in
                            zip(aids, sys_ranks, ref_ranks, ref_scores, answer_urls, answer_texts)
                        ]
                    }
                }
            elif self.config.schema == "bigbio_qa":
                yield qid, {
                    "id": qid,
                    "question_id": qid,
                    "document_id": qid,
                    "question": question_text,
                    "type": 'factoid',
                    "choices": [],
                    "context": '',
                    "answer": answer_texts,
                }
