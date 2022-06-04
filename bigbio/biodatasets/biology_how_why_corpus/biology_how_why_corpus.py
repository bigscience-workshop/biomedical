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
This dataset consists of 185 "how" and 193 "why" biology questions authored by a domain expert, with one or more gold 
answer passages identified in an undergraduate textbook. The expert was not constrained in any way during the 
annotation process, so gold answers might be smaller than a paragraph or span multiple paragraphs. This dataset was 
used for the question-answering system described in the paper “Discourse Complements Lexical Semantics for Non-factoid 
Answer Reranking” (ACL 2014).
"""

import os
import xml.dom.minidom as xml
from itertools import chain, count
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = False
_CITATION = """\
@inproceedings{,
  title={Discourse Complements Lexical Semantics for Non-factoid Answer Reranking},
  author={Peter Alexander Jansen and Mihai Surdeanu and Peter Clark},
  booktitle={ACL},
  year={2014}
}
"""

_DATASETNAME = "biology_how_why_corpus"

_DESCRIPTION = """\
This dataset consists of 185 "how" and 193 "why" biology questions authored by a domain expert, with one or more gold 
answer passages identified in an undergraduate textbook. The expert was not constrained in any way during the 
annotation process, so gold answers might be smaller than a paragraph or span multiple paragraphs. This dataset was 
used for the question-answering system described in the paper “Discourse Complements Lexical Semantics for Non-factoid 
Answer Reranking” (ACL 2014).
"""

_HOMEPAGE = "https://allenai.org/data/biology-how-why-corpus"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://ai2-public-datasets.s3.amazonaws.com/biology-how-why-corpus/BiologyHowWhyCorpus.tar",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class BiologyHowWhyCorpusDataset(datasets.GeneratorBasedBuilder):
    """This dataset consists of 185 "how" and 193 "why" biology questions."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="biology_how_why_corpus_source",
            version=SOURCE_VERSION,
            description="Biology How Why Corpus source schema",
            schema="source",
            subset_id="biology_how_why_corpus",
        ),
        BigBioConfig(
            name="biology_how_why_corpus_bigbio_qa",
            version=BIGBIO_VERSION,
            description="Biology How Why Corpus BigBio schema",
            schema="bigbio_qa",
            subset_id="biology_how_why_corpus",
        ),
    ]

    DEFAULT_CONFIG_NAME = "biology_how_why_corpus_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "answers": [
                        {
                            "justification": datasets.Value("string"),
                            "docid": datasets.Value("string"),
                            "sentences": datasets.Sequence(datasets.Value("int32")),
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
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "how_path": os.path.join(
                        data_dir, "BiologyHowWhyCorpus", "GoldStandardVulcanHOW.all.xml"
                    ),
                    "why_path": os.path.join(
                        data_dir, "BiologyHowWhyCorpus", "GoldStandardVulcanWHY.all.xml"
                    ),
                },
            ),
        ]

    def _generate_examples(self, how_path: str, why_path: str) -> Tuple[int, Dict]:

        uid = count(0)

        if self.config.schema == "source":
            for question in chain(
                self._parse_questions(how_path, "how"),
                self._parse_questions(why_path, "why"),
            ):
                yield next(uid), question

        elif self.config.schema == "bigbio_qa":
            for question in chain(
                self._parse_questions(how_path, "how"),
                self._parse_questions(why_path, "why"),
            ):
                for answer in question["answers"]:
                    id = next(uid)
                    yield id, {
                        "id": id,
                        "question_id": next(uid),
                        "document_id": answer["docid"],
                        "question": question["text"],
                        "type": question["type"],
                        "choices": [],
                        "context": "",
                        "answer": [answer["justification"]],
                    }

    def _parse_questions(self, path: str, type: str):
        collection = xml.parse(path).documentElement
        questions = collection.getElementsByTagName("question")
        for question in questions:
            text = question.getElementsByTagName("text")[0].childNodes[0].data
            answers = question.getElementsByTagName("answer")
            answers_ = []
            for answer in answers:
                justification = (
                    answer.getElementsByTagName("justification")[0].childNodes[0].data
                )
                docid = answer.getElementsByTagName("docid")[0].childNodes[0].data
                sentences = (
                    answer.getElementsByTagName("sentences")[0]
                    .childNodes[0]
                    .data.split(", ")
                )
                answers_.append(
                    {
                        "justification": justification,
                        "docid": docid,
                        "sentences": sentences,
                    }
                )
            yield {"text": text, "type": type, "answers": answers_}
