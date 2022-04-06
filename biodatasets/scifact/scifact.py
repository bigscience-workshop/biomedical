# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
#
# TODO: fill out the line below
# * <append your name and optionally your github handle here>
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
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.
When modifying it for your dataset, look for TODO items that offer specific instructions.
Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html
To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.
TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.
"""

import os
import json
from typing import List, Tuple
from itertools import chain

import datasets
from dataclasses import dataclass

from utils import schemas
from utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {David Wadden and Shanchuan Lin and Kyle Lo and Lucy Lu Wang and Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
  title     = {Fact or Fiction: Verifying Scientific Claims},
  year      = {2020},
  address   = {Online},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2020.emnlp-main.609},
  doi       = {10.18653/v1/2020.emnlp-main.609},
  pages     = {7534--7550},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "scifact"

_SOURCE_CORPUS_DESCRIPTION = """\
    SciFact is a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales.
    The task is to verify or refute claims using evidence from the abstracts.
    """

_BIGBIO_ENTAILMENT_RATIONALE_DESCRIPTION = """\
    SciFact is a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales.
    This task is the following: given a claim and an abstract, label that sentence with a 1/0 indicating if it is evidence (can be supporting or refuting). This corresponds to the second task outlined in Section 5 of the paper."
    """

_BIGBIO_ENTAILMENT_LABELPREDICTION_DESCRIPTION = """\
    SciFact is a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales.
    This task is the following: given a claim and several sentences of evidence (that either support or refute the claim), label that sentence with one of {REFUTES, SUPPORTS, NOINFO}. This corresponds to the third task outlined in Section 5 of the paper.
    """

_DESCRIPTION = {
    "scifact_source_corpus": _SOURCE_CORPUS_DESCRIPTION,
    "scifact_source_claims": _SOURCE_CORPUS_DESCRIPTION,
    "scifact_bigbio_entailment_rationale": _BIGBIO_ENTAILMENT_RATIONALE_DESCRIPTION,
    "scifact_bigbio_entailment_labelprediction": _BIGBIO_ENTAILMENT_LABELPREDICTION_DESCRIPTION,
}

_HOMEPAGE = "https://scifact.apps.allenai.org/"


_LICENSE = "CC BY-NC 2.0"

_URLS = {
    _DATASETNAME: "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    name: str = None
    version: str = None
    description: str = None
    schema: str = None
    subset_id: str = None


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class SciFact(datasets.GeneratorBasedBuilder):
    """
    SciFact is a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # TODO: For each dataset, implement Config for Source and BigBio;
    #  if dataset contains more than one subset (see examples/bioasq.py) implement for EACH of them. Each of them should contain:
    #  name: should be unique for each dataset config eg. bioasq10b_(source|bigbio)_[bigbioschema_name]
    #  version: option = (SOURCE_VERSION |BIGBIO_VERSION)
    #  description: one line description for the dataset
    #  schema: options = (source|bigbio_[schema_name]) [schema_name] =(kb,pairs, qa, text, test_to_text, entailment)
    #  subset_id: subset id is the canonical name for the dataset (eg. bioasq10b)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="scifact_source_corpus",
            version=SOURCE_VERSION,
            description="scifact source schema for the corpus config",
            schema="source",
            subset_id="scifact",
        ),
        BigBioConfig(
            name="scifact_source_claims",
            version=SOURCE_VERSION,
            description="scifact source schema for the claims config",
            schema="source",
            subset_id="scifact",
        ),
        BigBioConfig(
            name="scifact_bigbio_entailment_rationale",
            version=BIGBIO_VERSION,
            description="scifact BigBio text entailment schema for rationale task",
            schema="bigbio_entailment",
            subset_id="scifact",
        ),
        BigBioConfig(
            name="scifact_bigbio_entailment_labelprediction",
            version=BIGBIO_VERSION,
            description="scifact BigBio text entailment schema for label prediction task",
            schema="bigbio_entailment",
            subset_id="scifact",
        ),
    ]

    DEFAULT_CONFIG_NAME = "scifact_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            # https://huggingface.co/datasets/scifact/blob/main/scifact.py#L50

            if self.config.name == "scifact_source_corpus":
                features = datasets.Features(
                    {
                        "doc_id": datasets.Value("int32"),  # The document's S2ORC ID.
                        "title": datasets.Value("string"),  # The title.
                        "abstract": datasets.features.Sequence(
                            datasets.Value("string")
                        ),  # The abstract, written as a list of sentences.
                        "structured": datasets.Value(
                            "bool"
                        ),  # Indicator for whether this is a structured abstract.
                    }
                )
            elif self.config.name == "scifact_source_claims":
                features = datasets.Features(
                    {
                        "id": datasets.Value("int32"),  # An integer claim ID.
                        "claim": datasets.Value("string"),  # The text of the claim.
                        "evidence_doc_id": datasets.Value("string"),
                        "evidence_label": datasets.Value(
                            "string"
                        ),  # Label for the rationale.
                        "evidence_sentences": datasets.features.Sequence(
                            datasets.Value("int32")
                        ),  # Rationale sentences.
                        "cited_doc_ids": datasets.features.Sequence(
                            datasets.Value("int32")
                        ),  # The claim's "cited documents".
                    }
                )
            else:
                raise NotImplementedError(f"{self.config.name} config not implemented")

        elif self.config.schema == "bigbio_entailment":
            features = schemas.entailment.features

        else:
            raise NotImplementedError(f"{self.config.schema} schema not implemented")

        return datasets.DatasetInfo(
            description=_DESCRIPTION[self.config.name],
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        self.config.data_dir = dl_manager.download_and_extract(urls)

        if self.config.name == "scifact_source_corpus":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            self.config.data_dir, "data", "corpus.jsonl"
                        ),
                        "split": "train",
                    },
                ),
            ]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        self.config.data_dir, "data", "claims_train.jsonl"
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        self.config.data_dir, "data", "claims_test.jsonl"
                    ),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        self.config.data_dir, "data", "claims_dev.jsonl"
                    ),
                    "split": "dev",
                },
            ),
        ]


    def _source_generate_examples(self, filepath, split):
        # https://huggingface.co/datasets/scifact/blob/main/scifact.py#L136
        with open(filepath) as fp:
            for id_, row in enumerate(fp.readlines()):
                data = json.loads(row)
                if self.config.name == "scifact_source_corpus":
                    yield id_, {
                        "doc_id": int(data["doc_id"]),
                        "title": data["title"],
                        "abstract": data["abstract"],
                        "structured": data["structured"],
                    }
                elif self.config.name == "scifact_source_claims":
                    if split == "test":
                        yield id_, {
                            "id": data["id"],
                            "claim": data["claim"],
                            "evidence_doc_id": "",
                            "evidence_label": "",
                            "evidence_sentences": [],
                            "cited_doc_ids": [],
                        }
                    else:
                        evidences = data["evidence"]
                        if evidences:
                            for id1, doc_id in enumerate(evidences):
                                for id2, evidence in enumerate(evidences[doc_id]):
                                    yield str(id_) + "_" + str(id1) + "_" + str(id2), {
                                        "id": data["id"],
                                        "claim": data["claim"],
                                        "evidence_doc_id": doc_id,
                                        "evidence_label": evidence["label"],
                                        "evidence_sentences": evidence["sentences"],
                                        "cited_doc_ids": data.get("cited_doc_ids", []),
                                    }
                        else:
                            yield id_, {
                                "id": data["id"],
                                "claim": data["claim"],
                                "evidence_doc_id": "",
                                "evidence_label": "",
                                "evidence_sentences": [],
                                "cited_doc_ids": data.get("cited_doc_ids", []),
                            }

    def _bigbio_rationale_generate_examples(self, filepath, split, corpus_id2text):
        """
        Given a claim and sentence, decide if that sentence is a rationale (either supporting or contradicting) for the claim
        """
        with open(filepath) as fp:
            # Loop through each line of the file
            for line in fp.readlines():
                line = json.loads(line)
                claim = line["claim"]

                # test split doesn't have hypothesis or label
                if split == "test":
                    yield line["id"], {
                        "id": line["id"],
                        "premise": claim,
                        "hypothesis": "",
                        "label": "",
                    }
                    continue

                evidence = line["evidence"]
                line_id = str(line["id"])

                # Loop through each doc that is cited
                # Must take set because there are some with the same doc id multiple times
                for cited_doc_id in list(set(line[("cited_doc_ids")])):
                    rationale_sentence_ids = set()
                    if str(cited_doc_id) in evidence:

                        # this is a list of list of ints
                        rationale_sentence_ids = [
                            x["sentences"] for x in evidence[str(cited_doc_id)]
                        ]
                        rationale_sentence_ids = set(
                            list(chain(*rationale_sentence_ids))
                        )

                    # Loop through each sentence in the cited doc

                    for id3, sentence in enumerate(corpus_id2text[cited_doc_id]):
                        label = "rationale" if id3 in rationale_sentence_ids else "not_rationale"

                        # original line id, doc id, and sentence number
                        unique_id = (
                            f"{line_id.zfill(4)}_{cited_doc_id}_{str(id3).zfill(4)}"
                        )

                        yield unique_id, {
                            "id": unique_id,
                            "premise": claim,
                            "hypothesis": sentence,
                            "label": label,
                        }

    def _bigbio_rationale_generate_examples(self, filepath, split, corpus_id2text):
        """
        Given a claim and sentence, decide if that sentence is a rationale (support or contradict) for the claim
        """

    def _bigbio_generate_examples(self, filepath, split):

        corpus_id2text = {}
        with open(
            os.path.join(self.config.data_dir, "data", "corpus.jsonl")
        ) as corpus_fp:

            for line in corpus_fp.readlines():
                line = json.loads(line)
                corpus_id2text[line["doc_id"]] = line["abstract"]

        if self.config.name == "scifact_bigbio_entailment_rationale":
            return self._bigbio_rationale_generate_examples(
                filepath, split, corpus_id2text
            )
        elif self.config.name == "scifact_bigbio_entailment_labelprediction":
            return self._bigbio_labelprediction_generate_examples(
                filepath, split, corpus_id2text
            )

    def _generate_examples(self, filepath, split) -> Tuple[int, dict]:

        if self.config.name.startswith("scifact_source"):
            return self._source_generate_examples(filepath, split)

        elif "bigbio" in self.config.name:
            return self._bigbio_generate_examples(filepath, split)


if __name__ == "__main__":
    datasets.load_dataset(__file__)
