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
Relation Extraction corpus with multiple entity types (e.g., gene/protein, disease, chemical) and relation pairs (e.g., gene-disease; chemical-chemical), on a set of 600 PubMed articles
"""

import itertools
import os
from typing import Dict, List, Tuple

import datasets
from bioc import pubtator

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

# TODO: Add BibTeX citation
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
  @misc{https://doi.org/10.48550/arxiv.2204.04263,
  doi = {10.48550/ARXIV.2204.04263},

  url = {https://arxiv.org/abs/2204.04263},

  author = {Luo, Ling and Lai, Po-Ting and Wei, Chih-Hsuan and Arighi, Cecilia N and Lu, Zhiyong},

  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {BioRED: A Comprehensive Biomedical Relation Extraction Dataset},

  publisher = {arXiv},

  year = {2022},

  copyright = {Creative Commons Attribution 4.0 International}
}

"""

_DATASETNAME = "biored"

_DESCRIPTION = """Relation Extraction corpus with multiple entity types (e.g., gene/protein, disease, chemical) and relation pairs (e.g., gene-disease; chemical-chemical), on a set of 600 PubMed articles
"""

_HOMEPAGE = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

logger = datasets.utils.logging.get_logger(__name__)


class BioredDataset(datasets.GeneratorBasedBuilder):
    """Relation Extraction corpus with multiple entity types (e.g., gene/protein, disease, chemical) and relation pairs (e.g., gene-disease; chemical-chemical), on a set of 600 PubMed articles"""

    # For bigbio_kb, this dataset uses a naming convention as
    # uid_[title/abstract/relation/entity_id]_[entity/relation_uid]

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=_DATASETNAME + "_source",
            version=SOURCE_VERSION,
            description=_DATASETNAME + " source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        BigBioConfig(
            name=_DATASETNAME + "_bigbio_kb",
            version=BIGBIO_VERSION,
            description=_DATASETNAME + " BigBio schema",
            schema="bigbio_kb",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = _DATASETNAME + "_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "passages": [
                        {
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "entities": [
                        {
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "concept_id": datasets.Value("string"),
                            "semantic_type_id": datasets.Sequence(
                                datasets.Value("string")
                            ),
                        }
                    ],
                    "relations": [
                        {
                            "novel": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "concept_1": datasets.Value("string"),
                            "concept_2": datasets.Value("string"),
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
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "BioRED", "Train.PubTator"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "BioRED", "Test.PubTator"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "BioRED", "Dev.PubTator"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, "r", encoding="utf8") as fstream:
                for raw_document in self.generate_raw_docs(fstream):
                    document = self.parse_raw_doc(raw_document)
                    yield document["pmid"], document

        elif self.config.schema == "bigbio_kb":
            with open(filepath, "r", encoding="utf8") as fstream:
                uid = itertools.count(0)
                for raw_document in self.generate_raw_docs(fstream):
                    entities_in_doc = dict()
                    document = self.parse_raw_doc(raw_document)
                    pmid = document.pop("pmid")
                    document["id"] = str(next(uid))
                    document["document_id"] = pmid
                    entities_ = []
                    relations_ = []
                    for entity in document["entities"]:
                        temp_id = document["id"] + "_" + str(entity["concept_id"])
                        curr_entity_count = entities_in_doc.get(temp_id, 0)
                        entities_.append(
                            {
                                "id": temp_id + "_" + str(curr_entity_count),
                                "type": entity["semantic_type_id"],
                                "text": entity["text"],
                                "normalized": [],
                                "offsets": entity["offsets"],
                            }
                        )
                        entities_in_doc[temp_id] = curr_entity_count + 1
                    rel_uid = itertools.count(0)
                    for relation in document["relations"]:
                        relations_.append(
                            {
                                "id": document["id"]
                                + "_relation_"
                                + str(next(rel_uid)),
                                "type": relation["type"],
                                "arg1_id": document["id"]
                                + "_"
                                + str(relation["concept_1"])
                                + "_0",
                                "arg2_id": document["id"]
                                + "_"
                                + str(relation["concept_2"])
                                + "_0",
                                "normalized": [],
                            }
                        )
                    for passage in document["passages"]:
                        passage["id"] = document["id"] + "_" + passage["type"]
                    document["entities"] = entities_
                    document["relations"] = relations_
                    document["events"] = []
                    document["coreferences"] = []
                    yield document["document_id"], document

    def generate_raw_docs(self, fstream):
        """
        Given a filestream, this function yields documents from it
        """
        raw_document = []
        for line in fstream:
            if line.strip():
                raw_document.append(line.strip())
            elif raw_document:
                yield raw_document
                raw_document = []
        if raw_document:
            yield raw_document

    def parse_raw_doc(self, raw_doc):
        pmid, _, title = raw_doc[0].split("|")
        pmid = int(pmid)
        _, _, abstract = raw_doc[1].split("|")
        passages = [
            {"type": "title", "text": [title], "offsets": [[0, len(title)]]},
            {
                "type": "abstract",
                "text": [abstract],
                "offsets": [[len(title) + 1, len(title) + len(abstract) + 1]],
            },
        ]
        entities = []
        relations = []
        for line in raw_doc[2:]:
            mentions = line.split("\t")
            (_pmid, _type_ind, *rest) = mentions
            if _type_ind in [
                "Positive_Correlation",
                "Association",
                "Negative_Correlation",
                "Bind",
                "Conversion",
                "Cotreatment",
                "Cause",
                "Comparison",
                "Drug_Interaction",
            ]:
                # Relations handled here
                relation_type = _type_ind
                concept_1, concept_2, novel = rest
                relation = {
                    "type": relation_type,
                    "concept_1": concept_1,
                    "concept_2": concept_2,
                    "novel": novel,
                }
                relations.append(relation)
            elif _type_ind.isnumeric():
                # Entities handled here
                start_idx = _type_ind
                end_idx, mention, semantic_type_id, entity_ids = rest
                entity = [
                    {
                        "offsets": [[int(start_idx), int(end_idx)]],
                        "text": [mention],
                        "semantic_type_id": semantic_type_id.split(","),
                        "concept_id": entity_id,
                    }
                    for entity_id in entity_ids.split(",")
                ]
                entities.extend(entity)
            else:
                logger.warn(
                    f"Skipping annotation in Document ID: {_pmid}. Unexpected format"
                )
        return {
            "pmid": pmid,
            "passages": passages,
            "entities": entities,
            "relations": relations,
        }
