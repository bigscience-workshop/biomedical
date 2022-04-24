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
The IEPA benchmark PPI corpus is designed for relation extraction. It was created from 303 PubMed abstracts,
each of which contains a specific pair of co-occurring chemicals.
"""

# Comment from Author
# BigBio schema fixes offsets of entities to an offset where 0 is the start of the document.
# (In source offsets of entities start from 0 for each passage in document)
# Offsets of entities in source remain unchanged.

import xml.dom.minidom as xml
from typing import Dict, List, Tuple

import datasets

from biomed_datasets.utils import schemas
from biomed_datasets.utils.configs import BigBioConfig
from biomed_datasets.utils.constants import Tasks

_CITATION = """\
@ARTICLE{,
  title    = "Mining {MEDLINE}: abstracts, sentences, or phrases?",
  author   = "Ding, J and Berleant, D and Nettleton, D and Wurtele, E",
  journal  = "Pac Symp Biocomput",
  pages    = "326--337",
  year     =  2002,
  address  = "United States",
  language = "en"
}
"""

_DATASETNAME = "iepa"

_DESCRIPTION = """\
The IEPA benchmark PPI corpus is designed for relation extraction. It was created from 303 PubMed abstracts,
each of which contains a specific pair of co-occurring chemicals.
"""

_HOMEPAGE = "http://psb.stanford.edu/psb-online/proceedings/psb02/abstracts/p326.html"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/IEPA-train.xml",
        "test": "https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/IEPA-test.xml",
    },
}

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class IepaDataset(datasets.GeneratorBasedBuilder):
    """The IEPA benchmark PPI corpus is designed for relation extraction."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="iepa_source",
            version=SOURCE_VERSION,
            description="IEPA source schema",
            schema="source",
            subset_id="iepa",
        ),
        BigBioConfig(
            name="iepa_bigbio_kb",
            version=BIGBIO_VERSION,
            description="IEPA BigBio schema",
            schema="bigbio_kb",
            subset_id="iepa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "iepa_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "PMID": datasets.Value("string"),
                    "origID": datasets.Value("string"),
                    "sentences": [
                        {
                            "id": datasets.Value("string"),
                            "origID": datasets.Value("string"),
                            "offsets": [datasets.Value("int32")],
                            "text": datasets.Value("string"),
                            "entities": [
                                {
                                    "id": datasets.Value("string"),
                                    "origID": datasets.Value("string"),
                                    "text": datasets.Value("string"),
                                    "offsets": [datasets.Value("int32")],
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        collection = xml.parse(filepath).documentElement

        if self.config.schema == "source":
            for id, document in self._parse_documents(collection):
                yield id, document

        elif self.config.schema == "bigbio_kb":
            for id, document in self._parse_documents(collection):
                yield id, self._source_to_bigbio(document)

    def _parse_documents(self, collection):
        for document in collection.getElementsByTagName("document"):
            pmid_doc = self._strict_get_attribute(document, "PMID")
            id_doc = self._strict_get_attribute(document, "id")
            origID_doc = self._strict_get_attribute(document, "origID")
            sentences = []
            for sentence in document.getElementsByTagName("sentence"):
                offsets_sent = self._strict_get_attribute(sentence, "charOffset").split("-")
                id_sent = self._strict_get_attribute(sentence, "id")
                origID_sent = self._strict_get_attribute(sentence, "origID")
                text_sent = self._strict_get_attribute(sentence, "text")

                entities = []
                for entity in sentence.getElementsByTagName("entity"):
                    id_ent = self._strict_get_attribute(entity, "id")
                    origID_ent = self._strict_get_attribute(entity, "origID")
                    text_ent = self._strict_get_attribute(entity, "text")
                    offsets_ent = self._strict_get_attribute(entity, "charOffset").split("-")
                    entities.append({"id": id_ent, "origID": origID_ent, "text": text_ent, "offsets": offsets_ent})

                interactions = []
                for interaction in sentence.getElementsByTagName("interaction"):
                    id_int = self._strict_get_attribute(interaction, "id")
                    e1_int = self._strict_get_attribute(interaction, "e1")
                    e2_int = self._strict_get_attribute(interaction, "e2")
                    type_int = self._strict_get_attribute(interaction, "type")
                    interactions.append({"id": id_int, "e1": e1_int, "e2": e2_int, "type": type_int})

                sentences.append(
                    {
                        "id": id_sent,
                        "origID": origID_sent,
                        "offsets": offsets_sent,
                        "text": text_sent,
                        "entities": entities,
                        "interactions": interactions,
                    }
                )
            yield id_doc, {"id": id_doc, "PMID": pmid_doc, "origID": origID_doc, "sentences": sentences}

    def _strict_get_attribute(self, element, key):
        if element.hasAttribute(key):
            return element.getAttribute(key)
        else:
            raise ValueError(f"No such key exists in element: {element.tagName} {key}")

    def _source_to_bigbio(self, document_):
        document = {}
        document["id"] = document_["id"]
        document["document_id"] = document_["PMID"]

        passages = []
        entities = []
        relations = []
        for sentence_ in document_["sentences"]:
            for entity_ in sentence_["entities"]:
                entity_["type"] = ""
                entity_["normalized"] = []
                entity_.pop("origID")
                entity_["text"] = [entity_["text"]]
                entity_["offsets"] = [
                    [
                        int(sentence_["offsets"][0]) + int(entity_["offsets"][0]),
                        int(sentence_["offsets"][0]) + int(entity_["offsets"][1]),
                    ]
                ]
                entities.append(entity_)
            for relation_ in sentence_["interactions"]:
                relation_["arg1_id"] = relation_.pop("e1")
                relation_["arg2_id"] = relation_.pop("e2")
                relation_["normalized"] = []
                relations.append(relation_)

            sentence_.pop("entities")
            sentence_.pop("interactions")
            sentence_.pop("origID")
            sentence_["type"] = ""
            sentence_["text"] = [sentence_["text"]]
            sentence_["offsets"] = [sentence_["offsets"]]
            passages.append(sentence_)

        document["passages"] = passages
        document["entities"] = entities
        document["relations"] = relations
        document["events"] = []
        document["coreferences"] = []
        return document
