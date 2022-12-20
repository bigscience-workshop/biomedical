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

import xml.dom.minidom as xml
from itertools import count
from typing import BinaryIO, Dict, List, Tuple

import datasets

from .bigbiohub import BigBioConfig, Tasks, qa_features

_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = False
_CITATION = """\
@inproceedings{jansen-etal-2014-discourse,
    title = "Discourse Complements Lexical Semantics for Non-factoid Answer Reranking",
    author = "Jansen, Peter  and
      Surdeanu, Mihai  and
      Clark, Peter",
    booktitle = "Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jun,
    year = "2014",
    address = "Baltimore, Maryland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P14-1092",
    doi = "10.3115/v1/P14-1092",
    pages = "977--986",
}
"""

_DATASETNAME = "biology_how_why_corpus"
_DISPLAYNAME = "BiologyHowWhyCorpus"

_DESCRIPTION = """\
This dataset consists of 185 "how" and 193 "why" biology questions authored by a domain expert, with one or more gold 
answer passages identified in an undergraduate textbook. The expert was not constrained in any way during the 
annotation process, so gold answers might be smaller than a paragraph or span multiple paragraphs. This dataset was 
used for the question-answering system described in the paper “Discourse Complements Lexical Semantics for Non-factoid 
Answer Reranking” (ACL 2014).
"""

_HOMEPAGE = "https://allenai.org/data/biology-how-why-corpus"

_LICENSE = "License information unavailable"

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
            features = qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        archive_path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archive": dl_manager.iter_archive(archive_path),
                },
            ),
        ]

    def _generate_examples(self, archive: Tuple[str, BinaryIO]) -> Tuple[int, Dict]:

        uid = count(0)

        if self.config.schema == "source":
            for path, file in archive:
                question_type = path.split(".")[-3][-3:].lower()
                for question in self._parse_questions(file, question_type):
                    yield next(uid), question

        elif self.config.schema == "bigbio_qa":
            for path, file in archive:
                question_type = path.split(".")[-3][-3:].lower()
                for question in self._parse_questions(file, question_type):
                    for answer in question["answers"]:
                        guid = next(uid)
                        yield guid, {
                            "id": guid,
                            "question_id": next(uid),
                            "document_id": answer["docid"],
                            "question": question["text"],
                            "type": question["type"],
                            "choices": [],
                            "context": "",
                            "answer": [answer["justification"]],
                        }

    def _parse_questions(self, path_or_file: BinaryIO, question_type: str):
        collection = xml.parse(path_or_file).documentElement
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
            yield {"text": text, "type": question_type, "answers": answers_}
