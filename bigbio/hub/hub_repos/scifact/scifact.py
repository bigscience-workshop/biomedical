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
import os
from itertools import chain
from typing import Dict, List, Tuple

import datasets
from datasets import Value
import pandas as pd

from .bigbiohub import pairs_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ['English']
_PUBMED = False
_LOCAL = False
_CITATION = """\
@article{wadden2020fact,
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
_DISPLAYNAME = "SciFact"


_DESCRIPTION_BASE = """\
    SciFact is a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales.
    """

_SOURCE_CORPUS_DESCRIPTION = f"""\
    {_DESCRIPTION_BASE} This config has abstracts and document ids.
    """

_SOURCE_CLAIMS_DESCRIPTION = """\
    {_DESCRIPTION_BASE} This config connects the claims to the evidence and doc ids.
    """

_BIGBIO_PAIRS_RATIONALE_DESCRIPTION = """\
    {_DESCRIPTION_BASE} This task is the following: given a claim and a text span composed of one or more sentences from an abstract, predict a label from ("rationale", "not_rationale") indicating if the span is evidence (can be supporting or refuting) for the claim. This roughly corresponds to the second task outlined in Section 5 of the paper."
    """

_BIGBIO_PAIRS_LABELPREDICTION_DESCRIPTION = """\
    {_DESCRIPTION_BASE} This task is the following: given a claim and a text span composed of one or more sentences from an abstract, predict a label from ("SUPPORT", "NOINFO", "CONTRADICT") indicating if the span supports, provides no info, or contradicts the claim. This roughly corresponds to the thrid task outlined in Section 5 of the paper.
    """

_DESCRIPTION = {
    "scifact_corpus_source": _SOURCE_CORPUS_DESCRIPTION,
    "scifact_claims_source": _SOURCE_CLAIMS_DESCRIPTION,
    "scifact_rationale_bigbio_pairs": _BIGBIO_PAIRS_RATIONALE_DESCRIPTION,
    "scifact_labelprediction_bigbio_pairs": _BIGBIO_PAIRS_LABELPREDICTION_DESCRIPTION,
}

_HOMEPAGE = "https://scifact.apps.allenai.org/"


_LICENSE = 'Creative Commons Attribution Non Commercial 2.0 Generic'

_URLS = {
    _DATASETNAME: "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.TEXT_PAIRS_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class SciFact(datasets.GeneratorBasedBuilder):
    """
    SciFact is a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="scifact_corpus_source",
            version=SOURCE_VERSION,
            description="scifact source schema for the corpus config",
            schema="source",
            subset_id="scifact_corpus_source",
        ),
        BigBioConfig(
            name="scifact_claims_source",
            version=SOURCE_VERSION,
            description="scifact source schema for the claims config",
            schema="source",
            subset_id="scifact_claims_source",
        ),
        BigBioConfig(
            name="scifact_rationale_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="scifact BigBio text pairs classification schema for rationale task",
            schema="bigbio_pairs",
            subset_id="scifact_rationale_pairs",
        ),
        BigBioConfig(
            name="scifact_labelprediction_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="scifact BigBio text pairs classification schema for label prediction task",
            schema="bigbio_pairs",
            subset_id="scifact_labelprediction_pairs",
        ),
    ]

    DEFAULT_CONFIG_NAME = "scifact_claims_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            # modified from
            # https://huggingface.co/datasets/scifact/blob/main/scifact.py#L50

            if self.config.name == "scifact_corpus_source":
                features = datasets.Features(
                    {
                        "doc_id": Value("int32"),      # The document's S2ORC ID.
                        "title": Value("string"),      # The title.
                        "abstract": [Value("string")], # The abstract, written as a list of sentences.
                        "structured": Value("bool"),   # Indicator for whether this is a structured abstract.
                    }
                )

            elif self.config.name == "scifact_claims_source":
                features = datasets.Features(
                    {
                        "id": Value("int32"),  # An integer claim ID.
                        "claim": Value("string"),  # The text of the claim.
                        "evidences": [
                            {
                                "doc_id": Value("int32"),         # source doc_id for evidence
                                "sentence_ids": [Value("int32")], # sentence ids from doc_id
                                "label": Value("string"),         # SUPPORT or CONTRADICT
                            },
                        ],
                        "cited_doc_ids": [Value("int32")],   # The claim's "cited documents".
                    }
                )

            else:
                raise NotImplementedError(
                    f"{self.config.name} config not implemented"
                )

        elif self.config.schema == "bigbio_pairs":
            features = pairs_features

        else:
            raise NotImplementedError(f"{self.config.schema} schema not implemented")

        return datasets.DatasetInfo(
            description=_DESCRIPTION[self.config.name],
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        self.config.data_dir = dl_manager.download_and_extract(urls)

        if self.config.name == "scifact_corpus_source":
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

        # the test split is only returned in source schema
        # this is b/c it only has claims with no cited docs or evidence
        # the bigbio implementation of this dataset requires
        # cited docs or evidence to construct samples
        elif self.config.name == "scifact_claims_source":
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
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            self.config.data_dir, "data", "claims_dev.jsonl"
                        ),
                        "split": "dev",
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
            ]

        elif self.config.name in [
            "scifact_rationale_bigbio_pairs",
            "scifact_labelprediction_bigbio_pairs",
        ]:
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
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            self.config.data_dir, "data", "claims_dev.jsonl"
                        ),
                        "split": "dev",
                    },
                ),
            ]


    def _source_generate_examples(self, filepath, split) -> Tuple[str, Dict[str, str]]:

        # here we just read corpus.jsonl and return the abstracts
        if self.config.name == "scifact_corpus_source":
            with open(filepath) as fp:
                for id_, row in enumerate(fp.readlines()):
                    data = json.loads(row)
                    yield id_, {
                        "doc_id": int(data["doc_id"]),
                        "title": data["title"],
                        "abstract": data["abstract"],
                        "structured": data["structured"],
                    }

        # here we are reading one of claims_(train|dev|test).jsonl
        elif self.config.name == "scifact_claims_source":

            # claims_test.jsonl only has "id" and "claim" keys
            # claims_train.jsonl and claims_dev.jsonl sometimes have evidence
            with open(filepath) as fp:
                for id_, row in enumerate(fp.readlines()):
                    data = json.loads(row)
                    evidences_dict = data.get("evidence", {})
                    evidences_list = []
                    for doc_id, sent_lbl_list in evidences_dict.items():
                        for sent_lbl_dict in sent_lbl_list:
                            evidence = {
                                "doc_id": doc_id,
                                "sentence_ids": sent_lbl_dict["sentences"],
                                "label": sent_lbl_dict["label"],
                            }
                            evidences_list.append(evidence)

                    yield id_, {
                        "id": data["id"],
                        "claim": data["claim"],
                        "evidences": evidences_list,
                        "cited_doc_ids": data.get("cited_doc_ids", []),
                    }


    def _bigbio_generate_examples(self, filepath, split) -> Tuple[str, Dict[str, str]]:
        """
        Here we always create one sample per sentence group.
        Any sentence group in an evidence attribute will have
        a label in {"rationale"} for the rationale task or
        in {"SUPPORT", "CONTRADICT"} for the labelprediction task.
        All other sentences will have either a "not_rationale"
        label or a "NOINFO" label depending on the task.
        """

        # read corpus (one row per abstract)
        corpus_file_path = os.path.join(self.config.data_dir, "data", "corpus.jsonl")
        df_corpus = pd.read_json(corpus_file_path, lines=True)

        # create one row per sentence and create sentence index
        df_sents = df_corpus.explode('abstract')
        df_sents = df_sents.rename(columns={"abstract": "sentence"})
        df_sents['sent_num'] = df_sents.groupby('doc_id').transform('cumcount')
        df_sents['doc_sent_id'] = df_sents.apply(lambda x: f"{x['doc_id']}-{x['sent_num']}", axis=1)

        # read claims
        df_claims = pd.read_json(filepath, lines=True)


        # join claims to corpus
        for _, claim_row in df_claims.iterrows():

            evidence = claim_row['evidence']
            cited_doc_ids = set(claim_row['cited_doc_ids'])
            evidence_doc_ids = set([int(doc_id) for doc_id in evidence.keys()])

            # assert all evidence doc IDs are in cited_doc_ids
            assert len(evidence_doc_ids - cited_doc_ids) == 0

            # this will have all abstract sentences from cited docs
            df_claim_sents = df_sents[df_sents['doc_id'].isin(cited_doc_ids)]

            # create all sentence samples as NOINFO then fix
            noinfo_samples = {}
            for _, row in df_claim_sents.iterrows():
                sample = {
                    "claim": claim_row["claim"],
                    "claim_id": claim_row["id"],
                    "doc_id": row['doc_id'],
                    "sentence_ids": (row['sent_num'],),
                    "doc_sent_ids": (row['doc_sent_id'],),
                    "span": row['sentence'].strip(),
                    "label": "NOINFO",
                }
                noinfo_samples[sample["doc_sent_ids"]] = sample

            # create evidence samples and remove from noinfo samples as we go
            evidence_samples = []
            for doc_id_str, sent_lbl_list in evidence.items():
                doc_id = int(doc_id_str)

                for sent_lbl_dict in sent_lbl_list:
                    sent_ids = sent_lbl_dict['sentences']
                    doc_sent_ids = [f"{doc_id}-{sent_id}" for sent_id in sent_ids]
                    df_evi = df_claim_sents[df_claim_sents['doc_sent_id'].isin(doc_sent_ids)]

                    sample = {
                        "claim": claim_row["claim"],
                        "claim_id": claim_row["id"],
                        "doc_id": doc_id,
                        "sentence_ids": tuple(sent_ids),
                        "doc_sent_ids": tuple(doc_sent_ids),
                        "span": " ".join([el.strip() for el in df_evi["sentence"].values]),
                        "label": sent_lbl_dict["label"],
                    }
                    evidence_samples.append(sample)
                    for doc_sent_id in doc_sent_ids:
                        del noinfo_samples[(doc_sent_id,)]

            # combine all sample and put back in sentence order
            all_samples = evidence_samples + list(noinfo_samples.values())
            all_samples = sorted(all_samples, key=lambda x: (x['doc_id'], x['sentence_ids'][0]))

            # add a unique ID
            for _id, sample in enumerate(all_samples):
                sample["id"] = f"{_id}-{sample['claim_id']}-{sample['doc_id']}-{sample['sentence_ids'][0]}"

            RATIONALE_LABEL_MAP = {
                "SUPPORT": "rationale",
                "CONTRADICT": "rationale",
                "NOINFO": "not_rationale",
            }

            if self.config.name == "scifact_rationale_bigbio_pairs":
                for sample in all_samples:
                    yield sample['id'], {
                        "id": sample["id"],
                        "document_id": sample["doc_id"],
                        "text_1": sample["claim"],
                        "text_2": sample["span"],
                        "label": RATIONALE_LABEL_MAP[sample['label']],
                    }

            elif self.config.name == "scifact_labelprediction_bigbio_pairs":
                for sample in all_samples:
                    yield sample['id'], {
                        "id": sample["id"],
                        "document_id": sample["doc_id"],
                        "text_1": sample["claim"],
                        "text_2": sample["span"],
                        "label": sample['label'],
                    }

    def _generate_examples(self, filepath, split) -> Tuple[int, dict]:

        if "source" in self.config.name:
            for sample in self._source_generate_examples(filepath, split):
                yield sample

        elif "bigbio" in self.config.name:
            for sample in self._bigbio_generate_examples(filepath, split):
                yield sample
