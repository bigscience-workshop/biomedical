# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
The BioCreative VI Chemical-Protein interaction dataset identifies entities of
chemicals and proteins and their likely relation to one other. Compounds are
generally agonists (activators) or antagonists (inhibitors) of proteins. The
script loads dataset in bigbio schema (using knowledgebase schema: schemas/kb)
AND/OR source (default) schema
"""
from pathlib import Path
from typing import Dict, Any, List, Union

import datasets
import json

from .bigbiohub import qa_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = False

_CITATION = """\
@inproceedings{goodwin2020overview,
  title={Overview of the 2020 epidemic question answering track},
  author={Goodwin, TRAVIS R and Demner-Fushman, Dina and Lo, Kyle and Wang, Lucy Lu and Hersh, WILLIAM R and Dang, HT and Soboroff, Ian M},
  booktitle={Text Analysis Conference},
  year={2020}
}
"""
_DESCRIPTION = """\
In response to the COVID-19 pandemic, the Epidemic Question Answering (EPIC-QA) track
challenges teams to develop systems capable of automatically answering ad-hoc questions about the disease COVID-19, its causal virus SARS-CoV-2, 
related corona viruses, and the recommended response to the pandemic.
"""

_DATASETNAME = "epic_qa"
_DISPLAYNAME = "EPIC-QA"

_HOMEPAGE = "https://bionlp.nlm.nih.gov/epic_qa/"

_LICENSE = "DUA"

_BASE = "https://bionlp.nlm.nih.gov/epic_qa/data/epic_qa_"

_URLs = {
    "epic_qa_research_source": _BASE + "cord_2020-10-22-split-corrected.tar.gz",
    "epic_qa_research_bigbio_qa": _BASE + "cord_2020-10-22-split-corrected.tar.gz",
    "epic_qa_consumer_source": _BASE + "consumer_2020-11-02.tar.gz",
    "epic_qa_consumer_bigbio_qa": _BASE + "consumer_2020-11-02.tar.gz",
}

_DIRS = {
    "research": "2020-10-22-new",
    "consumer": "epic_qa_consumer_2020_11-02",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class EpicQaDataset(datasets.GeneratorBasedBuilder):
    """
    EpicQA

    In response to the COVID-19 pandemic, the Epidemic Question
    Answering (EPIC-QA) track challenges teams to develop systems
    capable of automatically answering ad-hoc questions about the
    disease COVID-19, its causal virus SARS-CoV-2, related corona
    viruses, and the recommended response to the pandemic.
    """

    DEFAULT_CONFIG_NAME = "epic_qa_research_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    for subset in ["research", "consumer"]:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"epic_qa_{subset}_source",
                version=SOURCE_VERSION,
                description=f"EpicQA Source schema for the {subset.capitalize()} Articles subset",
                schema="source",
                subset_id=f"epic_qa_{subset}",
            )
        )
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"epic_qa_{subset}_bigbio_qa",
                version=BIGBIO_VERSION,
                description=f"EpicQA BigBio schema for the {subset.capitalize()} Articles subset",
                schema="bigbio_qa",
                subset_id=f"epic_qa_{subset}",
            )
        )

    def _info(self):
        if self.config.name == "epic_qa_consumer_source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "metadata": {
                        "url": datasets.Value("string"),
                    },
                    "question": datasets.Value("string"),
                    "context_id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "section": datasets.Value("string"),
                    "answer": [
                        {
                            "id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                        }
                    ],
                }
            )
        elif self.config.name == "epic_qa_research_source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "metadata": {
                        "urls": datasets.Sequence(datasets.Value("string")),
                        "authors": datasets.Value("string"),
                        "full_text_path": datasets.Value("string"),
                    },
                    "question": datasets.Value("string"),
                    "context_id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "section": datasets.Value("string"),
                    "answer": [
                        {
                            "id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                        }
                    ],
                }
            )
        else:
            features = qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(data_dir), "split": "train"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str):
        """Yields examples as (key, example) tuples."""

        if self.config.subset_id.endswith("research"):
            subset_name = "research"
        elif self.config.subset_id.endswith("consumer"):
            subset_name = "consumer"
        else:
            raise ValueError(f"subset_id {self.config.subset_id} does not end with 'research' or 'consumer'")

        sub_directory = _DIRS[subset_name]

        sub_directory_path = filepath / sub_directory
        files_path_glob = sub_directory_path.rglob("*.json")

        uid = 0
        for file_full_path in files_path_glob:
            with open(file_full_path, encoding="utf-8") as file:
                try:
                    doc: Dict = json.load(file)
                except json.JSONDecodeError:
                    print("File %s is malformed and cannot be decoded.", file_full_path.name)
                    continue

            document_id: str = doc.pop("document_id")
            metadata: Dict[str, Any] = doc.pop("metadata")
            question: str = metadata.pop("title")
            contexts: List[Dict[str, Any]] = doc.pop("contexts")

            for context in contexts:
                if self.config.name.endswith("source"):
                    source_example = self._to_source_example(
                        document_id=document_id,
                        metadata=metadata,
                        question=question,
                        context=context,
                    )
                    yield uid, source_example

                # Generate the BigBio schema
                else:
                    bigbio_example = self._to_bigbio_example(
                        document_id=document_id,
                        question=question,
                        context=context,
                    )
                    yield uid, bigbio_example

                uid += 1

    def _to_source_example(
        self,
        document_id: str,
        metadata: Dict[str, Any],
        question: str,
        context: Dict[str, Any],
    ) -> Dict:

        answers: List[Dict[str, Union[str, int]]] = []
        for sentence in context["sentences"]:
            start = sentence["start"]
            end = sentence["end"]
            answer = {
                "id": sentence["sentence_id"],
                "text": context["text"][start:end],
                "start": start,
                "end": end,
            }
            answers.append(answer)

        source_example = {
            "document_id": document_id,
            "metadata": metadata,
            "question": question,
            "context_id": context["context_id"],
            "context": context["text"],
            "section": context["section"],
            "answer": answers,
        }

        return source_example

    def _to_bigbio_example(
        self,
        document_id: str,
        question: str,
        context: Dict[str, Any],
    ) -> Dict:

        answers: List[str] = []
        for sentence in context["sentences"]:
            start = sentence["start"]
            end = sentence["end"]
            answer = context["text"][start:end]
            answers.append(answer)

        bigbio_example = {
            "id": context["context_id"],
            "question_id": context["context_id"],
            "document_id": document_id,
            "question": question,
            "type": "list",
            "choices": [],
            "context": context["text"],
            "answer": answers,
        }
        return bigbio_example
