# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and Simon Ott, github: nomisto
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
The LLL05 challenge task is to learn rules to extract protein/gene interactions from biology abstracts from the Medline
bibliography database. The goal of the challenge is to test the ability of the participating IE systems to identify the
interactions and the gene/proteins that interact. The participants will test their IE patterns on a test set with the
aim of extracting the correct agent and target.The challenge focuses on information extraction of gene interactions in
Bacillus subtilis. Extracting gene interaction is the most popular event IE task in biology. Bacillus subtilis (Bs) is
a model bacterium and many papers have been published on direct gene interactions involved in sporulation. The gene
interactions are generally mentioned in the abstract and the full text of the paper is not needed. Extracting gene
interaction means, extracting the agent (proteins) and the target (genes) of all couples of genic interactions from
sentences.
"""

import itertools as it
from typing import List

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@misc{mohan2019medmentions,
      title={MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts},
      author={Sunil Mohan and Donghui Li},
      year={2019},
      eprint={1902.09476},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "medmentions"

_DESCRIPTION = """\
The LLL05 challenge task is to learn rules to extract protein/gene interactions from biology abstracts from the Medline
bibliography database. The goal of the challenge is to test the ability of the participating IE systems to identify the
interactions and the gene/proteins that interact. The participants will test their IE patterns on a test set with the
aim of extracting the correct agent and target.The challenge focuses on information extraction of gene interactions in
Bacillus subtilis. Extracting gene interaction is the most popular event IE task in biology. Bacillus subtilis (Bs) is
a model bacterium and many papers have been published on direct gene interactions involved in sporulation. The gene
interactions are generally mentioned in the abstract and the full text of the paper is not needed. Extracting gene
interaction means, extracting the agent (proteins) and the target (genes) of all couples of genic interactions from
sentences.
"""

_HOMEPAGE = "http://genome.jouy.inra.fr/texte/LLLchallenge"

_LICENSE = "?"

_URLS = {
    _DATASETNAME: [
        "http://genome.jouy.inra.fr/texte/LLLchallenge/data/LLLChalenge05/data/train/task2/genic_interaction_linguistic_data.txt",  # noqa
        "http://genome.jouy.inra.fr/texte/LLLchallenge/data/LLLChalenge05/data/train/task2/genic_interaction_linguistic_data_coref.txt",  # noqa
        "http://genome.jouy.inra.fr/texte/LLLchallenge/data/LLLChalenge05/data/test/task2/enriched_test_data.txt",  # noqa
    ]
}

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class LLLDataset(datasets.GeneratorBasedBuilder):
    """LLL dataset for gene interection extraction (RE)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="lll_source",
            version=SOURCE_VERSION,
            description="LLL source schema",
            schema="source",
            subset_id="lll",
        ),
        BigBioConfig(
            name="lll_bigbio_kb",
            version=BIGBIO_VERSION,
            description="LLL BigBio schema",
            schema="bigbio_kb",
            subset_id="lll",
        ),
    ]

    DEFAULT_CONFIG_NAME = "lll_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "words": [
                        {
                            "id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "genic_interactions": [
                        {
                            "ref_id1": datasets.Value("string"),
                            "ref_id2": datasets.Value("string"),
                        }
                    ],
                    "agents": [
                        {
                            "ref_id": datasets.Value("string"),
                        }
                    ],
                    "targets": [
                        {
                            "ref_id": datasets.Value("string"),
                        }
                    ],
                    "lemmas": [{"ref_id": datasets.Value("string"), "lemma": datasets.Value("string")}],
                    "syntactic_relations": [
                        {
                            "type": datasets.Value("string"),
                            "ref_id1": datasets.Value("string"),
                            "ref_id2": datasets.Value("string"),
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        urls = _URLS[_DATASETNAME]
        train_path, train_coref_path, test_path = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_paths": [train_path, train_coref_path], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_paths": [test_path], "split": "test"},
            ),
        ]

    def _generate_examples(self, data_paths, split):

        if self.config.schema == "source":
            for path in data_paths:
                with open(path, encoding="utf8") as documents:
                    for document in self._generate_parsed_documents(documents, split):
                        yield document["id"], document

        elif self.config.schema == "bigbio_kb":
            uid = it.count(0)
            for path in data_paths:
                with open(path, encoding="utf8") as documents:
                    for document in self._generate_parsed_documents(documents, split):
                        document_ = {}
                        document_["id"] = next(uid)
                        document_["document_id"] = document["id"]

                        document_["passages"] = [
                            {
                                "id": next(uid),
                                "type": None,
                                "text": [document["sentence"]],
                                "offsets": [[0, len(document["sentence"])]],
                            }
                        ]

                        document_["entities"] = [
                            {
                                "id": f"{document_['id']}-{word['id']}",
                                "type": None,
                                "text": [word["text"]],
                                "offsets": [word["offsets"]],
                                "normalized": [
                                    {
                                        "db_name": None,
                                        "db_id": None,
                                    }
                                ],
                            }
                            for word in document["words"]
                        ]

                        document_["relations"] = [
                            {
                                "id": next(uid),
                                "type": "genic_interaction",
                                "arg1_id": f"{document_['id']}-{relation['ref_id1']}",
                                "arg2_id": f"{document_['id']}-{relation['ref_id2']}",
                                "normalized": [
                                    {
                                        "db_name": None,
                                        "db_id": None,
                                    }
                                ],
                            }
                            for relation in document["genic_interactions"]
                        ]

                        document_["events"] = []
                        document_["coreferences"] = []
                        yield document_["document_id"], document_

    def _generate_parsed_documents(self, fstream, split):
        for raw_document in self._generate_raw_documents(fstream):
            yield self._parse_document(raw_document, split)

    def _generate_raw_documents(self, fstream):
        raw_document = []
        for line in fstream:
            if "%" in line:
                continue
            elif line.strip():
                raw_document.append(line.strip())
            elif raw_document:
                if raw_document:
                    yield raw_document
                raw_document = []
        # needed for last document
        if raw_document:
            yield raw_document

    def _parse_document(self, raw_document, split):
        document = {}
        for line in raw_document:
            key, value = line.split("\t", 1)
            if key in ["ID", "sentence"]:
                document[key.lower()] = value
            elif key in ["words", "genic_interactions", "agents", "targets", "lemmas", "syntactic_relations"]:
                document[key.lower()] = self._parse_elements(value, key)
            else:
                raise NotImplementedError()

        # Needed as testset does not contain agents, targets and genic_interactions (dataset was part of a challenge)
        if split == "test":
            document.setdefault("genic_interactions", [])
            document.setdefault("agents", [])
            document.setdefault("targets", [])

        return document

    def _parse_elements(self, values, type):
        return [self._parse_element(atom, type) for atom in values.split("\t")]

    def _parse_element(self, atom, type):
        # Sorry for that abomination, parses the arguments from atoms like rel(arg1, ..., argn)
        args = atom.split("(", 1)[1][:-1].split(",")
        if type == "words":
            return {"id": args[0], "text": args[1].strip("'"), "offsets": [args[2], args[3]]}
        elif type == "genic_interactions":
            return {"ref_id1": args[0], "ref_id2": args[1]}
        elif type == "agents":
            return {"ref_id": args[0]}
        elif type == "targets":
            return {"ref_id": args[0]}
        elif type == "lemmas":
            return {"ref_id": args[0], "lemma": args[1]}
        elif type == "syntactic_relations":
            return {"type": args[0], "ref_id1": args[1], "ref_id2": args[2]}
