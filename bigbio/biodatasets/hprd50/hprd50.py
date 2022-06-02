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
HPRD50 is a dataset of randomly selected, hand-annotated abstracts of biomedical papers
referenced by the Human Protein Reference Database (HPRD). It is parsed in XML format,
splitting each abstract into sentences, and in each sentence there may be entities and
interactions between those entities. In this particular dataset, entities are all
proteins and interactions are thus protein-protein interactions.

Moreover, all entities are normalized to the HPRD database. These normalized terms are
stored in each entity's 'type' attribute in the source XML. This means the dataset can
determine e.g. that "Janus kinase 2" and "Jak2" are referencing the same normalized
entity.

Because the dataset contains entities and relations, it is suitable for Named Entity
Recognition and Relation Extraction.
"""

import os
from glob import glob
from typing import Dict, List, Tuple
from xml.etree import ElementTree

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

# TODO: Add BibTeX citation
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{,
  author    = {Katrin Fundel, Robert Kuffner, Ralf Zimmer},
  title     = {RelEx-Relation extraction using dependency parse trees},
  journal   = {Bioinformatics},
  volume    = {23},
  year      = {2007},
  url       = {https://academic.oup.com/bioinformatics/article/23/3/365/236564},
  doi       = {https://doi.org/10.1093/bioinformatics/btl616},
}
"""

_DATASETNAME = "hprd50"

_DESCRIPTION = """\
HPRD50 is a dataset of randomly selected, hand-annotated abstracts of biomedical papers
referenced by the Human Protein Reference Database (HPRD). It is parsed in XML format,
splitting each abstract into sentences, and in each sentence there may be entities and
interactions between those entities. In this particular dataset, entities are all
proteins and interactions are thus protein-protein interactions.

Moreover, all entities are normalized to the HPRD database. These normalized terms are
stored in each entity's 'type' attribute in the source XML. This means the dataset can
determine e.g. that "Janus kinase 2" and "Jak2" are referencing the same normalized
entity.

Because the dataset contains entities and relations, it is suitable for Named Entity
Recognition and Relation Extraction.
"""

_HOMEPAGE = ""

_LICENSE = ""

_URLS = {
    _DATASETNAME: "https://github.com/metalrt/ppi-dataset/zipball/master",
}

_SUPPORTED_TASKS = [
    Tasks.RELATION_EXTRACTION,
    Tasks.NAMED_ENTITY_RECOGNITION,
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


def parse_xml_source(document_trees):
    entries = []
    for doc in document_trees:
        document = {
            "id": doc.get("id"),
            "origId": doc.get("origId"),
            "set": doc.get("test"),
            "sentences": [],
        }
        for s in doc.findall("sentence"):
            sentence = {
                "id": s.get("id"),
                "origId": s.get("origId"),
                "charOffset": s.get("charOffset"),
                "text": s.get("text"),
                "entities": [],
                "interactions": [],
            }

            for e in s.findall("entity"):
                entity = {
                    "id": e.get("id"),
                    "origId": e.get("origId"),
                    "charOffset": e.get("charOffset"),
                    "text": e.get("text"),
                    "type": e.get("type"),
                }

                sentence["entities"].append(entity)

            for i in s.findall("interaction"):
                interaction = {
                    "id": i.get("id"),
                    "e1": i.get("e1"),
                    "e2": i.get("e2"),
                    "type": i.get("type"),
                }
                sentence["interactions"].append(interaction)

            document["sentences"].append(sentence)

        entries.append(document)
    return entries


def parse_xml_bigbio_kb(document_trees):
    entries = []
    for doc in document_trees:
        document = {
            "id": doc.get("id"),
            "document_id": doc.get("origId"),
            "passages": [],
            "entities": [],
            "relations": [],
            "events": [],
            "coreferences": [],
        }
        for s in doc.findall("sentence"):

            offset = s.get("charOffset").split("-")
            start = int(offset[0])
            end = int(offset[1])

            passage = {
                "id": s.get("id"),
                "type": "sentence",
                "text": [s.get("text")],
                "offsets": [[start, end]],
            }

            document["passages"].append(passage)

            for e in s.findall("entity"):

                offset = e.get("charOffset").split("-")
                start = int(offset[0])
                end = int(offset[1])

                entity = {
                    "id": e.get("id"),
                    "text": [e.get("text")],
                    "offsets": [[start, end]],
                    "type": "protein",
                    "normalized": [{"db_name": "HPRD", "db_id": e.get("type")}],
                }

                document["entities"].append(entity)

            for i in s.findall("interaction"):
                relation = {
                    "id": i.get("id"),
                    "arg1_id": i.get("e1"),
                    "arg2_id": i.get("e2"),
                    "type": i.get("type"),
                    "normalized": [],
                }
                document["relations"].append(relation)

        entries.append(document)
    return entries


class HPRD50Dataset(datasets.GeneratorBasedBuilder):
    """
    HPRD50 is a dataset of randomly selected, hand-annotated abstracts of biomedical papers
    referenced by the Human Protein Reference Database (HPRD). It is parsed in XML format,
    splitting each abstract into sentences, and in each sentence there may be entities and
    interactions between those entities. In this particular dataset, entities are all
    proteins and interactions are thus protein-protein interactions.

    Moreover, all entities are normalized to the HPRD database. These normalized terms are
    stored in each entity's 'type' attribute in the source XML. This means the dataset can
    determine e.g. that "Janus kinase 2" and "Jak2" are referencing the same normalized
    entity.

    Because the dataset contains entities and relations, it is suitable for Named Entity
    Recognition and Relation Extraction.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="hprd50_source",
            version=SOURCE_VERSION,
            description="hprd50 source schema",
            schema="source",
            subset_id="hprd50",
        ),
        BigBioConfig(
            name="hprd50_bigbio_kb",
            version=BIGBIO_VERSION,
            description="hprd50 BigBio schema",
            schema="bigbio_kb",
            subset_id="hprd50",
        ),
    ]

    DEFAULT_CONFIG_NAME = "hprd50_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "origId": datasets.Value("string"),
                    "set": datasets.Value("string"),
                    "sentences": [
                        {
                            "id": datasets.Value("string"),
                            "origId": datasets.Value("string"),
                            "charOffset": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "entities": [
                                {
                                    "id": datasets.Value("string"),
                                    "origId": datasets.Value("string"),
                                    "charOffset": datasets.Value("string"),
                                    "text": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                }
                            ],
                            "interactions": [
                                {
                                    "id": datasets.Value("string"),
                                    "e1": datasets.Value("string"),
                                    "e2": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                }
                            ],
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
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        # Files are actually a few levels down, under this subdirectory, and
        # intermediate directory names get hashed so this is the easiest way to find it.
        data_dir = glob(f"{data_dir}/**/csv_output")[0]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "HPRD50-train.xml"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "HPRD50-test.xml"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as f:
            content = f.read()

        tree = ElementTree.fromstring(content)
        documents = tree.findall("document")

        if self.config.schema == "source":
            entries = parse_xml_source(documents)
            for key, example in enumerate(entries):
                yield key, example

        elif self.config.schema == "bigbio_kb":
            entries = parse_xml_bigbio_kb(documents)
            for key, example in enumerate(entries):
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
