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

import json
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

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

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION, Tasks.RELATION_EXTRACTION, Tasks.COREFERENCE_RESOLUTION]

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
                            "label": datasets.Value("int32")
                        }
                    ], 
                    "url": datasets.Value("string"), 
                    "text": datasets.Value("string"), 
                    "entities": [
                        {
                            "is_state": datasets.Value("bool"), 
                            "label": datasets.Value("string"), 
                            "names": [{
                                "text": datasets.Value("string"),
                                "is_mentioned": datasets.Value("bool"), 
                                "mentions": datasets.Sequence([datasets.Value("int32")])
                            }], 
                            "grounding": [{
                                "comment": datasets.Value("string"), 
                                "entrez_gene": datasets.Value("string"),
                                "source": datasets.Value("string"), 
                                "link": datasets.Value("string"),
                                "hgnc_symbol": datasets.Value("string"),
                                "organism": datasets.Value("string"),
                            }], 
                            "is_mentioned": datasets.Value("bool"),
                            "is_mutant": datasets.Value("bool"),
                        }
                    ],
                    "_line_": datasets.Value("int32"), 
                    "id": datasets.Value("string")
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
                yield key, self._source_to_kb(example)

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
            assert (new_key not in values), "New key already in values"
            values[new_key] = key
            list_.append(values)
        return list_

    def _source_to_kb(self, example):
        document_ = {
            "id": example["id"],
            "document_id": example["paperid"],
            "passages": [
                {
                    "id": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "text": example["text"],
                    "offsets": [[0,len(example["text"])]],
                }
            ],
            "entities": [
                {
                    "id": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "text": datasets.Sequence(datasets.Value("string")),
                    "offsets": datasets.Sequence([datasets.Value("int32")]),
                    "normalized": [
                        {
                            "db_name": datasets.Value("string"),
                            "db_id": datasets.Value("string"),
                        }
                    ],
                }
            ],
            "events": [
                {
                    "id": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    # refers to the text_bound_annotation of the trigger
                    "trigger": {
                        "text": datasets.Sequence(datasets.Value("string")),
                        "offsets": datasets.Sequence([datasets.Value("int32")]),
                    },
                    "arguments": [
                        {
                            "role": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                        }
                    ],
                }
            ],
            "coreferences": [
                {
                    "id": datasets.Value("string"),
                    "entity_ids": datasets.Sequence(datasets.Value("string")),
                }
            ],
            "relations": [
                {
                    "id": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "arg1_id": datasets.Value("string"),
                    "arg2_id": datasets.Value("string"),
                    "normalized": [
                        {
                            "db_name": datasets.Value("string"),
                            "db_id": datasets.Value("string"),
                        }
                    ],
                }
            ],
        }



        features = datasets.Features(
            {
                "paperid": datasets.Value("string"), 
                "interactions": [
                    {
                        "participants": datasets.Sequence(datasets.Value("int32")), 
                        "type": datasets.Value("string"), 
                        "implicit": datasets.Value("bool"), 
                        "label": datasets.Value("int32")
                    }
                ], 
                "url": datasets.Value("string"), 
                "text": datasets.Value("string"), 
                "entities": [
                    {
                        "is_state": datasets.Value("bool"), 
                        "label": datasets.Value("string"), 
                        "names": [{
                            "text": datasets.Value("string"),
                            "is_mentioned": datasets.Value("bool"), 
                            "mentions": datasets.Sequence([datasets.Value("int32")])
                        }], 
                        "grounding": [{
                            "comment": datasets.Value("string"), 
                            "entrez_gene": datasets.Value("string"),
                            "source": datasets.Value("string"), 
                            "link": datasets.Value("string"),
                            "hgnc_symbol": datasets.Value("string"),
                            "organism": datasets.Value("string"),
                        }], 
                        "is_mentioned": datasets.Value("bool"),
                        "is_mutant": datasets.Value("bool"),
                    }
                ],
                "_line_": datasets.Value("int32"), 
                "id": datasets.Value("string")
            }
        )

    
