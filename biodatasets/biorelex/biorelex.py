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
BioRelEx is a biological relation extraction dataset. Version 1.0 contains 2010 annotated sentences that describe
binding interactions between various biological entities (proteins, chemicals, etc.). 1405 sentences are for training,
another 201 sentences are for validation. They are publicly available at https://github.com/YerevaNN/BioRelEx/releases.
Another 404 sentences are for testing which are kept private for at this Codalab competition https:
//competitions.codalab.org/competitions/20468. All sentences contain words "bind", "bound" or "binding". For every
sentence we provide: 1) Complete annotations of all biological entities that appear in the sentence 2) Entity types (32
types) and grounding information for most of the proteins and families (links to uniprot, interpro and other databases)
3) Coreference between entities in the same sentence (e.g. abbreviations and synonyms) 4) Binding interactions between
the annotated entities 5) Binding interaction types: positive, negative (A does not bind B) and neutral (A may bind to
B)
"""

import itertools as it
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import datasets

from biomed_datasets.utils import schemas
from biomed_datasets.utils.configs import BigBioConfig
from biomed_datasets.utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{,
    title = "{B}io{R}el{E}x 1.0: Biological Relation Extraction Benchmark",
    author = "Khachatrian, Hrant  and
      Nersisyan, Lilit  and
      Hambardzumyan, Karen  and
      Galstyan, Tigran  and
      Hakobyan, Anna  and
      Arakelyan, Arsen  and
      Rzhetsky, Andrey  and
      Galstyan, Aram",
    booktitle = "Proceedings of the 18th BioNLP Workshop and Shared Task",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5019",
    doi = "10.18653/v1/W19-5019",
    pages = "176--190"
}
"""

_DATASETNAME = "biorelex"

_DESCRIPTION = """\
BioRelEx is a biological relation extraction dataset. Version 1.0 contains 2010 annotated sentences that describe
binding interactions between various biological entities (proteins, chemicals, etc.). 1405 sentences are for training,
another 201 sentences are for validation. They are publicly available at https://github.com/YerevaNN/BioRelEx/releases.
Another 404 sentences are for testing which are kept private for at this Codalab competition https:
//competitions.codalab.org/competitions/20468. All sentences contain words "bind", "bound" or "binding". For every
sentence we provide: 1) Complete annotations of all biological entities that appear in the sentence 2) Entity types (32
types) and grounding information for most of the proteins and families (links to uniprot, interpro and other databases)
3) Coreference between entities in the same sentence (e.g. abbreviations and synonyms) 4) Binding interactions between
the annotated entities 5) Binding interaction types: positive, negative (A does not bind B) and neutral (A may bind to
B)
"""

_HOMEPAGE = "https://github.com/YerevaNN/BioRelEx"

_LICENSE = "Unknown"

_URLS = {
    _DATASETNAME: {
        "train": "https://github.com/YerevaNN/BioRelEx/releases/download/1.0alpha7/1.0alpha7.train.json",
        "dev": "https://github.com/YerevaNN/BioRelEx/releases/download/1.0alpha7/1.0alpha7.dev.json",
    },
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
    Tasks.RELATION_EXTRACTION,
    Tasks.COREFERENCE_RESOLUTION,
]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class BioRelExDataset(datasets.GeneratorBasedBuilder):
    """BioRelEx is a biological relation extraction dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="biorelex_source",
            version=SOURCE_VERSION,
            description="BioRelEx source schema",
            schema="source",
            subset_id="biorelex",
        ),
        BigBioConfig(
            name="biorelex_bigbio_kb",
            version=BIGBIO_VERSION,
            description="BioRelEx BigBio schema",
            schema="bigbio_kb",
            subset_id="biorelex",
        ),
    ]

    DEFAULT_CONFIG_NAME = "biorelex_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "paperid": datasets.Value("string"),
                    "interactions": [
                        {
                            "participants": datasets.Sequence(datasets.Value("int32")),
                            "type": datasets.Value("string"),
                            "implicit": datasets.Value("bool"),
                            "label": datasets.Value("int32"),
                        }
                    ],
                    "url": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "is_state": datasets.Value("bool"),
                            "label": datasets.Value("string"),
                            "names": [
                                {
                                    "text": datasets.Value("string"),
                                    "is_mentioned": datasets.Value("bool"),
                                    "mentions": datasets.Sequence([datasets.Value("int32")]),
                                }
                            ],
                            "grounding": [
                                {
                                    "comment": datasets.Value("string"),
                                    "entrez_gene": datasets.Value("string"),
                                    "source": datasets.Value("string"),
                                    "link": datasets.Value("string"),
                                    "hgnc_symbol": datasets.Value("string"),
                                    "organism": datasets.Value("string"),
                                }
                            ],
                            "is_mentioned": datasets.Value("bool"),
                            "is_mutant": datasets.Value("bool"),
                        }
                    ],
                    "_line_": datasets.Value("int32"),
                    "id": datasets.Value("string"),
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
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["dev"],
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)
        data = self._prep(data)

        if self.config.schema == "source":
            for key, example in enumerate(data):
                yield key, example

        elif self.config.schema == "bigbio_kb":
            for key, example in enumerate(data):
                example_ = self._source_to_kb(example)
                yield key, example_

    def _prep(self, data):
        for example in data:
            for entity in example["entities"]:
                entity["names"] = self._json_dict_to_list(entity["names"], "text")
                if entity["grounding"] is None:
                    entity["grounding"] = []
                else:
                    entity["grounding"] = [entity["grounding"]]
        return data

    def _json_dict_to_list(self, json, new_key):
        list_ = []
        for key, values in json.items():
            assert isinstance(values, dict), "Child element is not a dict"
            assert new_key not in values, "New key already in values"
            values[new_key] = key
            list_.append(values)
        return list_

    def _source_to_kb(self, example):
        example_id = example["id"]
        entities_, corefs_, ref_id_map = self._get_entities(example_id, example["entities"])
        relations_ = self._get_relations(example_id, ref_id_map, example["interactions"])

        document_ = {
            "id": example_id,
            "document_id": example["paperid"],
            "passages": [
                {
                    "id": example_id + ".sent",
                    "type": "sentence",
                    "text": [example["text"]],
                    "offsets": [[0, len(example["text"])]],
                }
            ],
            "entities": entities_,
            "coreferences": corefs_,
            "relations": relations_,
            "events": [],
        }
        return document_

    def _get_entities(self, example_id, entities):
        entities_ = []
        corefs_ = []

        eid = it.count(0)
        cid = it.count(0)
        # dictionary mapping the original ref ids (indexes of entities) for relations
        org_rel_ref_id_2_kb_entity_id = defaultdict(list)

        for relation_ref_id, entity in enumerate(entities):

            # get normalization for entities
            normalized_ = self._get_normalizations(entity)

            # create entity for each synonym
            coref_eids_ = []
            for names in entity["names"]:
                for id, mention in enumerate(names["mentions"]):
                    entity_id = example_id + ".ent" + str(next(eid)) + "_" + str(id)
                    org_rel_ref_id_2_kb_entity_id[relation_ref_id].append(entity_id)
                    coref_eids_.append(entity_id)
                    entities_.append(
                        {
                            "id": entity_id,
                            "type": entity["label"],
                            "text": [names["text"]],
                            "offsets": [mention],
                            "normalized": normalized_,
                        }
                    )

            # create coreferences
            coref_id = example_id + ".coref" + str(next(cid))
            corefs_.append(
                {
                    "id": coref_id,
                    "entity_ids": coref_eids_,
                }
            )
        return entities_, corefs_, org_rel_ref_id_2_kb_entity_id

    def _get_normalizations(self, entity):
        normalized_ = []
        if entity["grounding"]:
            assert len(entity["grounding"]) == 1
            if entity["grounding"][0]["entrez_gene"] != "NA":
                normalized_.append({"db_name": "NCBI gene", "db_id": entity["grounding"][0]["entrez_gene"]})
            if entity["grounding"][0]["hgnc_symbol"] != "NA":
                normalized_.append({"db_name": "hgnc", "db_id": entity["grounding"][0]["hgnc_symbol"]})

            # maybe parse some other ids?
            source = entity["grounding"][0]["source"]
            if (
                source != "NCBI gene" and source != "https://www.genenames.org/data/genegroup/"
            ):  # NCBI gene is same as entrez
                normalized_.append(
                    self._parse_id_from_link(entity["grounding"][0]["link"], entity["grounding"][0]["source"])
                )
        return normalized_

    def _get_relations(self, example_id, org_rel_ref_id_2_kb_entity_id, interactions):
        rid = it.count(0)
        relations_ = []
        for interaction in interactions:
            rel_id = example_id + ".rel" + str(next(rid))
            assert len(interaction["participants"]) == 2

            subjects = org_rel_ref_id_2_kb_entity_id[interaction["participants"][0]]
            objects = org_rel_ref_id_2_kb_entity_id[interaction["participants"][1]]

            for s in subjects:
                for o in objects:
                    relations_.append(
                        {
                            "id": rel_id + "s" + s + ".o" + o,
                            "type": interaction["type"],
                            "arg1_id": s,
                            "arg2_id": o,
                            "normalized": [],
                        }
                    )
        return relations_

    def _parse_id_from_link(self, link, source):
        source_template_map = {
            "uniprot": ["https://www.uniprot.org/uniprot/"],
            "pubchem:compound": ["https://pubchem.ncbi.nlm.nih.gov/compound/"],
            "pubchem:substance": ["https://pubchem.ncbi.nlm.nih.gov/substance/"],
            "pfam": ["https://pfam.xfam.org/family/", "http://pfam.xfam.org/family/"],
            "interpro": ["http://www.ebi.ac.uk/interpro/entry/", "https://www.ebi.ac.uk/interpro/entry/"],
            "DrugBank": ["https://www.drugbank.ca/drugs/"],
        }

        # fix exceptions manually
        if source == "https://enzyme.expasy.org/EC/2.5.1.18" and link == source:
            return {"db_name": "intenz", "db_id": "2.5.1.18"}
        elif source == "https://www.genome.jp/kegg-bin/show_pathway?map=ko04120" and link == source:
            return {"db_name": "kegg", "db_id": "ko04120"}
        elif source == "https://www.genome.jp/dbget-bin/www_bget?enzyme+2.7.11.1" and link == source:
            return {"db_name": "intenz", "db_id": "2.7.11.1"}
        elif source == "http://www.chemspider.com/Chemical-Structure.7995676.html" and link == source:
            return {"db_name": "chemspider", "db_id": "7995676"}
        elif source == "intenz":
            id = link.split("=")[0]
            return {"db_name": source, "db_id": id}
        else:
            link_templates = source_template_map[source]
            for template in link_templates:
                if link.startswith(template):
                    id = link.replace(template, "")
                    id = id.split("?")[0]
                    assert "/" not in id
                    return {"db_name": source, "db_id": id}

            assert False, f"No template found for {link}, choices: {repr(link_templates)}"
