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
The compound-protein relationship (CPI) dataset consists of 2,613 sentences from abstracts containing
annotations of proteins, small molecules, and their relationships. For further information see:
https://pubmed.ncbi.nlm.nih.gov/32126064/ and https://github.com/KerstenDoering/CPI-Pipeline
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, Tuple

import datasets

from .bigbiohub import kb_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ['English']
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{doring2020automated,
  title={Automated recognition of functional compound-protein relationships in literature},
  author={D{\"o}ring, Kersten and Qaseem, Ammar and Becer, Michael and Li, Jianyu and Mishra, Pankaj and Gao, Mingjie and Kirchner, Pascal and Sauter, Florian and Telukunta, Kiran K and Moumbock, Aur{\'e}lien FA and others},
  journal={Plos one},
  volume={15},
  number={3},
  pages={e0220925},
  year={2020},
  publisher={Public Library of Science San Francisco, CA USA}
}
"""

_DATASETNAME = "cpi"
_DISPLAYNAME = "CPI"

_DESCRIPTION = """\
The compound-protein relationship (CPI) dataset consists of 2,613 sentences from abstracts containing \
annotations of proteins, small molecules, and their relationships
"""

_HOMEPAGE = "https://github.com/KerstenDoering/CPI-Pipeline"

_LICENSE = 'ISC License'

_URLS = {
    "CPI": "https://github.com/KerstenDoering/CPI-Pipeline/raw/master/data_sets/xml/CPI-DS.xml",
    "CPI_IV": "https://github.com/KerstenDoering/CPI-Pipeline/raw/master/data_sets/xml/CPI-DS_IV.xml",
    "CPI_NIV": "https://github.com/KerstenDoering/CPI-Pipeline/raw/master/data_sets/xml/CPI-DS_IV.xml",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.2"
_BIGBIO_VERSION = "1.0.0"


class CpiDataset(datasets.GeneratorBasedBuilder):
    """The compound-protein relationship (CPI) dataset"""

    ENTITY_TYPE_TO_DB_NAME = {"compound": "PubChem", "protein": "UniProt"}

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="cpi_source",
            version=SOURCE_VERSION,
            description="CPI source schema",
            schema="source",
            subset_id="cpi",
        ),
        BigBioConfig(
            name="cpi_iv_source",
            version=SOURCE_VERSION,
            description="CPI source schema - subset with interaction verbs",
            schema="source",
            subset_id="cpi_iv",
        ),
        BigBioConfig(
            name="cpi_niv_source",
            version=SOURCE_VERSION,
            description="CPI source schema - subset without interaction verbs",
            schema="source",
            subset_id="cpi_niv",
        ),
        BigBioConfig(
            name="cpi_bigbio_kb",
            version=BIGBIO_VERSION,
            description="CPI BigBio schema",
            schema="bigbio_kb",
            subset_id="cpi",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cpi_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "document_orig_id": datasets.Value("string"),
                    "sentences": [
                        {
                            "sentence_id": datasets.Value("string"),
                            "sentence_orig_id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "entities": [
                                {
                                    "entity_id": datasets.Value("string"),
                                    "entity_orig_id": datasets.Sequence(datasets.Value("string")),
                                    "type": datasets.Value("string"),
                                    "offset": datasets.Sequence(datasets.Value("int32")),
                                    "text": datasets.Value("string"),
                                }
                            ],
                            "pairs": [
                                {
                                    "pair_id": datasets.Value("string"),
                                    "e1": datasets.Value("string"),
                                    "e2": datasets.Value("string"),
                                    "interaction": datasets.Value("bool"),
                                }
                            ],
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Distinguish based on the subset id (cpi, cpi_iv, cpi_niv) which file to load
        subset_url = _URLS[self.config.subset_id.upper()]
        subset_file = dl_manager.download_and_extract(subset_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"subset_file": subset_file},
            )
        ]

    def _generate_examples(self, subset_file: Path) -> Iterator[Tuple[str, Dict]]:
        if self.config.schema == "source":
            for doc_id, document in self._read_source_examples(subset_file):
                yield doc_id, document

        elif self.config.name == "cpi_bigbio_kb":
            # Note: The sentences in a CPI document does not (necessarily) occur consecutive in
            # the original publication. Nevertheless, in this implementation we capture all sentences
            # of a document in one kb-schema document to explicitly model documents.

            # Transform each source-schema document to kb-schema document
            for doc_id, source_document in self._read_source_examples(subset_file):
                sentence_offset = 0
                passages = []
                entities = []
                relations = []

                # Transform all sentences to kb-schema sentences
                for source_sentence in source_document["sentences"]:
                    text = source_sentence["text"]
                    passages.append(
                        {
                            "id": source_sentence["sentence_id"],
                            "text": [text],
                            "offsets": [[sentence_offset + 0, sentence_offset + len(text)]],
                            "type": "",
                        }
                    )

                    # Transform source-schema entities to kb-schema entities
                    for source_entity in source_sentence["entities"]:
                        db_name = self.ENTITY_TYPE_TO_DB_NAME[source_entity["type"]]

                        entity_offset = source_entity["offset"]
                        entity_offset = [sentence_offset + entity_offset[0], sentence_offset + entity_offset[1]]

                        entities.append(
                            {
                                "id": source_entity["entity_id"],
                                "type": source_entity["type"],
                                "text": [source_entity["text"]],
                                "offsets": [entity_offset],
                                "normalized": [
                                    {"db_name": db_name, "db_id": db_id} for db_id in source_entity["entity_orig_id"]
                                ],
                            }
                        )

                    # Transform source-schema pairs to kb-schema relations
                    for source_pair in source_sentence["pairs"]:
                        # Ignore pairs that are annotated to be not in a relationship!
                        if not source_pair["interaction"]:
                            continue

                        relations.append(
                            {
                                "id": source_pair["pair_id"],
                                "type": "compound-protein-interaction",
                                "arg1_id": source_pair["e1"],
                                "arg2_id": source_pair["e2"],
                                "normalized": [],
                            }
                        )

                    sentence_offset += len(text) + 1

                kb_document = {
                    "id": source_document["document_id"],
                    "document_id": source_document["document_orig_id"],
                    "passages": passages,
                    "entities": entities,
                    "relations": relations,
                    "events": [],
                    "coreferences": [],
                }

                yield source_document["document_id"], kb_document

    def _read_source_examples(self, input_file: Path) -> Iterator[Tuple[str, Dict]]:
        """
        Reads all instances of the given input file and parses them into the source format.
        """
        root = ET.parse(input_file)
        for document in root.iter("document"):
            sentences = []
            for sentence in document.iter("sentence"):
                entities = []
                for entity in sentence.iter("entity"):
                    char_offsets = entity.attrib["charOffset"].split("-")
                    start, end = int(char_offsets[0]), int(char_offsets[1])

                    entities.append(
                        {
                            "entity_id": entity.attrib["id"],
                            "entity_orig_id": entity.attrib["origId"].split(","),
                            "type": entity.attrib["type"],
                            "text": entity.attrib["text"],
                            "offset": [start, end],
                        }
                    )

                pairs = []
                for pair in sentence.iter("pair"):
                    pairs.append(
                        {
                            "pair_id": pair.attrib["id"],
                            "e1": pair.attrib["e1"],
                            "e2": pair.attrib["e2"],
                            "interaction": pair.attrib["interaction"].lower() == "true",
                        }
                    )

                sentences.append(
                    {
                        "sentence_id": sentence.attrib["id"],
                        "sentence_orig_id": sentence.attrib["origId"],
                        "text": sentence.attrib["text"],
                        "entities": entities,
                        "pairs": pairs,
                    }
                )

            document_dict = {
                "document_id": document.attrib["id"],
                "document_orig_id": document.attrib["origId"],
                "sentences": sentences,
            }

            yield document.attrib["id"], document_dict
