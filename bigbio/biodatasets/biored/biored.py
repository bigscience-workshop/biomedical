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
Relation Extraction corpus with multiple entity types (e.g., gene/protein,
disease, chemical) and relation pairs (e.g., gene-disease; chemical-chemical),
on a set of 600 PubMed articles
"""

import itertools
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import datasets
from bioc import pubtator

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{DBLP:journals/corr/abs-2204-04263,
  author    = {Ling Luo and
               Po{-}Ting Lai and
               Chih{-}Hsuan Wei and
               Cecilia N. Arighi and
               Zhiyong Lu},
  title     = {BioRED: {A} Comprehensive Biomedical Relation Extraction Dataset},
  journal   = {CoRR},
  volume    = {abs/2204.04263},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2204.04263},
  doi       = {10.48550/arXiv.2204.04263},
  eprinttype = {arXiv},
  eprint    = {2204.04263},
  timestamp = {Wed, 11 May 2022 15:24:37 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2204-04263.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "biored"
_DISPLAYNAME = "BioRED"

_DESCRIPTION = """\
Relation Extraction corpus with multiple entity types (e.g., gene/protein,
disease, chemical) and relation pairs (e.g., gene-disease; chemical-chemical),
on a set of 600 PubMed articles
"""

_HOMEPAGE = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

logger = datasets.utils.logging.get_logger(__name__)


class BioredDataset(datasets.GeneratorBasedBuilder):
    """Relation Extraction corpus with multiple entity types (e.g., gene/protein, disease, chemical) and
    relation pairs (e.g., gene-disease; chemical-chemical), on a set of 600 PubMed articles"""

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

    TYPE_TO_DATABASE = {
        "CellLine": "Cellosaurus",
        "ChemicalEntity": "MESH",
        "DiseaseOrPhenotypicFeature": "MESH",  # Some diseases are normalized to OMIM (~ handled by special rules)
        "GeneOrGeneProduct": "NCBIGene",
        "OrganismTaxon": "NCBITaxon",
        "SequenceVariant": "dbSNP",  # Not all variants are normalized to dbSNP (~ handled by special rules)
    }

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
                            "semantic_type_id": datasets.Value("string"),
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
                    document = self.parse_raw_doc(raw_document)
                    pmid = str(document.pop("pmid"))
                    document["id"] = str(next(uid))
                    document["document_id"] = pmid

                    # Parse entities
                    entities = []
                    entity_id_to_mentions = defaultdict(list)  # Maps database ids to mention ids
                    for i, entity in enumerate(document["entities"]):
                        internal_id = pmid + "_" + str(i)

                        # Some entities are normalized to multiple database ids, therefore we
                        # may have multiple identifiers per mention
                        normalized_entity_ids = []
                        for database_id in entity["concept_id"].split(","):
                            database_id = database_id.strip()
                            entity_type = entity["semantic_type_id"]

                            # First check special db_name and database id assignment rules
                            if entity_type == "DiseaseOrPhenotypicFeature" and database_id.lower().startswith("omim"):
                                db_name = "OMIM"
                                database_id = database_id.split(":")[-1]
                            elif entity_type == "SequenceVariant" and not database_id.startswith("rs"):
                                db_name = "custom"

                            # If no special rule applies -> just take the default db_name for the entity type
                            else:
                                db_name = self.TYPE_TO_DATABASE[entity_type]

                            normalized_entity_ids.append({"db_name": db_name, "db_id": database_id})
                            entity_id_to_mentions[database_id].append(internal_id)

                        entities.append(
                            {
                                "id": internal_id,
                                "type": entity_type,
                                "text": entity["text"],
                                "normalized": normalized_entity_ids,
                                "offsets": entity["offsets"],
                            }
                        )

                    # BioRed provides abstract-level annotations for entity-linked relation pairs rather than
                    # materializing links between all surface form mentions of relation. For example document 11009181
                    # in train has (Positive_Correlation, D007980, D004409). Analogous to BC5CDR we enumerate all
                    # mention pairs concerning the entities in the triple.
                    relations = []
                    rel_uid = itertools.count(0)
                    for relation in document["relations"]:
                        head_mentions = entity_id_to_mentions[str(relation["concept_1"])]
                        tail_mentions = entity_id_to_mentions[str(relation["concept_2"])]

                        for head, tail in itertools.product(head_mentions, tail_mentions):
                            relations.append(
                                {
                                    "id": document["id"] + "_relation_" + str(next(rel_uid)),
                                    "type": relation["type"],
                                    "arg1_id": head,
                                    "arg2_id": tail,
                                    "normalized": [],
                                }
                            )

                    for passage in document["passages"]:
                        passage["id"] = document["id"] + "_" + passage["type"]

                    document["entities"] = entities
                    document["relations"] = relations
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
                entities.append(
                    {
                        "offsets": [[int(start_idx), int(end_idx)]],
                        "text": [mention],
                        "semantic_type_id": semantic_type_id,
                        "concept_id": entity_ids,
                    }
                )
            else:
                logger.warn(f"Skipping annotation in Document ID: {_pmid}. Unexpected format")
        return {
            "pmid": pmid,
            "passages": passages,
            "entities": entities,
            "relations": relations,
        }
