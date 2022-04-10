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

import datasets

from utils import schemas
from utils.constants import Tasks
from utils.configs import BigBioConfig

_CITATION = """\
Unknown
"""

_DESCRIPTION = """\
In response to the COVID-19 pandemic, the Epidemic Question
Answering (EPIC-QA) track challenges teams to develop systems
capable of automatically answering ad-hoc questions about the
disease COVID-19, its causal virus SARS-CoV-2, related corona
viruses, and the recommended response to the pandemic.
The 2020 EPIC-QA track involves two tasks:

Task A
    Expert QA: In Task A, teams are provided with a set of
    questions asked by experts and are asked to provide a
    ranked list of expert-level answers to each question.
    In Task A, answers should provide information that is
    useful to researchers, scientists, or clinicians.

Task B
    Consumer QA: In Task B, teams are provided with a set
    of questions asked by consumers and are asked to provide
    a ranked list of consumer-friendly answers to each question.
    In Task B, answers should be understandable by the general
    public.

While each task will have its own set of questions, many of the
questions will overlap. This is by design, so that the collection
can be used to explore whether the same approaches or systems can
account for different types of users.
"""

_DATASETNAME = "EpicQA"

_HOMEPAGE = "https://bionlp.nlm.nih.gov/epic_qa/#collection"

_LICENSE = "DUA-NC"

_BASE = "https://bionlp.nlm.nih.gov/epic_qa/data/epic_qa_"

_URLs = {
    "epicqa_research_source": _BASE + "cord_2020-10-22-split-corrected.tar.gz",
    "epicqa_research_bigbio_qa": _BASE + "cord_2020-10-22-split-corrected.tar.gz",
    
    "epicqa_consumer_source": _BASE + "consumer_2020-11-02.tar.gz",
    "epicqa_consumer_bigbio_qa": _BASE + "consumer_2020-11-02.tar.gz",
}

_DIRS = {
    "research" : [
        "2020-10-22-new/",
    ],
    "consumer" : [
        "epic_qa_consumer_2020_11-02/ask_science-2020-10-29/",
        "epic_qa_consumer_2020_11-02/ccns-trec/",
        "epic_qa_consumer_2020_11-02/chqa-2020-10-09/",
    ],
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

class EpicQaDataset(datasets.GeneratorBasedBuilder):
    """EpicQA"""

    DEFAULT_CONFIG_NAME = "epicqa_research_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [

        BigBioConfig(
            name="epicqa_research_source",
            version=SOURCE_VERSION,
            description="EpicQA source schema for the Research Articles subset",
            schema="source",
            subset_id="epicqa_research",
        ),

        BigBioConfig(
            name="epicqa_consumer_source",
            version=SOURCE_VERSION,
            description="EpicQA source schema for the Consumer Articles subset",
            schema="source",
            subset_id="epicqa_consumer",
        ),

        BigBioConfig(
            name="epicqa_research_bigbio_qa",
            version=BIGBIO_VERSION,
            description="EpicQA BigBio schema for the Research Articles subset",
            schema="bigbio_qa",
            subset_id="epicqa_research",
        ),

        BigBioConfig(
            name="epicqa_consumer_bigbio_qa",
            version=BIGBIO_VERSION,
            description="EpicQA BigBio schema for the Consumer Articles subset",
            schema="bigbio_qa",
            subset_id="epicqa_consumer",
        ),

    ]

    def _info(self):

        if self.config.name == "epicqa_consumer_source":

            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "metadata": {
                        "title": datasets.Value("string"),
                        "url": datasets.Value("string"),
                    },
                    "contexts": [
                        {
                            "section": datasets.Value("string"),
                            "context_id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "sentences": [
                                {
                                    "sentence_id": datasets.Value("string"),
                                    "start": datasets.Value("int32"),
                                    "end": datasets.Value("int32"),
                                },
                            ],
                        }
                    ],
                }
            )
        
        elif self.config.name == "epicqa_research_source":

            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "metadata": {
                        "title": datasets.Value("string"),
                        "authors": datasets.Value("string"),
                        "urls": datasets.Sequence(datasets.Value("string")),
                        "full_text_path": datasets.Value("string"),
                    },
                    "contexts": [
                        {
                            "section": datasets.Value("string"),
                            "context_id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "sentences": [
                                {
                                    "sentence_id": datasets.Value("string"),
                                    "start": datasets.Value("int32"),
                                    "end": datasets.Value("int32"),
                                },
                            ],
                        }
                    ],
                }
            )

        # Simplified schema for QA tasks
        else:

            features = schemas.qa_features

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
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""

        if self.config.subset_id.endswith("research"):
            subset_name = "research"
        elif self.config.subset_id.endswith("consumer"):
            subset_name = "consumer"
        else:
            raise ValueError(f"subset_id {self.config.subset_id} does not end with 'research' or 'consumer'")

        uid = 0

        # For each sub directory
        for sub_dir in _DIRS[subset_name]:

            # Get the JSON file
            for f_path in os.listdir(filepath + "/" + sub_dir):

                # Check if JSON file
                if ".json" not in f_path:
                    continue

                # Get the JSON file path
                full_path = filepath + "/" + sub_dir + f_path

                # Read the JSON file content
                with open(full_path, encoding="utf-8") as file:

                    # Parse it and add it to the data array
                    doc = json.load(file)

                    # Generate source schema for the consumers articles
                    if self.config.name == "epicqa_consumer_source":

                        yield uid, {
                            "document_id": doc["document_id"],
                            "metadata": {
                                "title": doc["metadata"]["title"],
                                "url": doc["metadata"]["url"],
                            },
                            "contexts": [
                                {
                                    "section": c["section"],
                                    "context_id": c["context_id"],
                                    "text": c["text"],
                                    "sentences": [
                                        {
                                            "sentence_id": s["sentence_id"],
                                            "start": s["start"],
                                            "end": s["end"],
                                        }
                                        for s in c["sentences"]
                                    ],
                                }
                                for c in doc["contexts"]
                            ],
                        }
                    
                        uid += 1

                    # Generate source schema for the research articles
                    elif self.config.name == "epicqa_research_source":

                        yield uid, {
                            "document_id": doc["document_id"],
                            "metadata": {
                                "title": doc["metadata"]["title"],
                                "authors": doc["metadata"]["authors"],
                                "urls": doc["metadata"]["urls"],
                                "full_text_path": doc["metadata"]["full_text_path"],
                            },
                            "contexts": [
                                {
                                    "section": c["section"],
                                    "context_id": c["context_id"],
                                    "text": c["text"],
                                    "sentences": [
                                        {
                                            "sentence_id": s["sentence_id"],
                                            "start": s["start"],
                                            "end": s["end"],
                                        }
                                        for s in c["sentences"]
                                    ],
                                }
                                for c in doc["contexts"]
                            ],
                        }

                        uid += 1

                    # Generate the BigBio schema
                    else:

                        contexts = []

                        # For each contexts
                        for c in doc["contexts"]:

                            spans = []

                            # For each span                            
                            for s in c["sentences"]:

                                spans.append(
                                    c["text"][int(s["start"]):int(s["end"])]
                                )

                            yield uid, {
                                "id": c["context_id"],
                                "document_id": doc["document_id"],
                                "question_id": c["context_id"],
                                "question": doc["metadata"]["title"],
                                "type": "list",
                                "context": c["text"],
                                "answer": spans,
                                "choices": [],
                            }

                            uid += 1
