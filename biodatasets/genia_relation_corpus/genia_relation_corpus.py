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
The extraction of various relations stated to hold between biomolecular entities is one of the most frequently
addressed information extraction tasks in domain studies. Typical relation extraction targets involve protein-protein
interactions or gene regulatory relations. However, in the GENIA corpus, such associations involving change in the
state or properties of biomolecules are captured in the event annotation.

The GENIA corpus relation annotation aims to complement the event annotation of the corpus by capturing (primarily)
static relations, relations such as part-of that hold between entities without (necessarily) involving change.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from utils import parsing, schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@inproceedings{pyysalo-etal-2009-static,
    title = "Static Relations: a Piece in the Biomedical Information Extraction Puzzle",
    author = "Pyysalo, Sampo  and
      Ohta, Tomoko  and
      Kim, Jin-Dong  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the {B}io{NLP} 2009 Workshop",
    month = jun,
    year = "2009",
    address = "Boulder, Colorado",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W09-1301",
    pages = "1--9",
}

@article{article,
author = {Ohta, Tomoko and Pyysalo, Sampo and Kim, Jin-Dong and Tsujii, Jun'ichi},
year = {2010},
month = {10},
pages = {917-28},
title = {A reevaluation of biomedical named entity - term relations},
volume = {8},
journal = {Journal of bioinformatics and computational biology},
doi = {10.1142/S0219720010005014}
}

@MISC{Hoehndorf_applyingontology,
    author = {Robert Hoehndorf and Axel-cyrille Ngonga Ngomo and Sampo Pyysalo and Tomoko Ohta and Anika Oellrich and
    Dietrich Rebholz-schuhmann},
    title = {Applying ontology design patterns to the implementation of relations in GENIA},
    year = {}
}
"""

_DATASETNAME = "genia_relation_corpus"

_DESCRIPTION = """\
The extraction of various relations stated to hold between biomolecular entities is one of the most frequently
addressed information extraction tasks in domain studies. Typical relation extraction targets involve protein-protein
interactions or gene regulatory relations. However, in the GENIA corpus, such associations involving change in the
state or properties of biomolecules are captured in the event annotation.

The GENIA corpus relation annotation aims to complement the event annotation of the corpus by capturing (primarily)
static relations, relations such as part-of that hold between entities without (necessarily) involving change.
"""

_HOMEPAGE = "http://www.geniaproject.org/genia-corpus/relation-corpus"

_LICENSE = """GENIA Project License for Annotated Corpora"""

_URLS = {
    _DATASETNAME: {
        "train": "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_training_data.tar.gz",
        "validation": "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_development_data.tar.gz",
        "test": "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_test_data.tar.gz",
    },
}

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

_ENTITY_TYPES = ["Protein"]


class GeniaRelationCorpusDataset(datasets.GeneratorBasedBuilder):
    """GENIA Relation corpus."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="genia_relation_corpus_source",
            version=SOURCE_VERSION,
            description="genia_relation_corpus source schema",
            schema="source",
            subset_id="genia_relation_corpus",
        ),
        BigBioConfig(
            name="genia_relation_corpus_bigbio_kb",
            version=BIGBIO_VERSION,
            description="genia_relation_corpus BigBio schema",
            schema="bigbio_kb",
            subset_id="genia_relation_corpus",
        ),
    ]

    DEFAULT_CONFIG_NAME = "genia_relation_corpus_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "relations": [  # R line in brat
                        {
                            "id": datasets.Value("string"),
                            "head": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "tail": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "type": datasets.Value("string"),
                        }
                    ],
                    "equivalences": [  # Equiv line in brat
                        {
                            "id": datasets.Value("string"),
                            "ref_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                },
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
                name=split,
                gen_kwargs={
                    "data_dir": data_dir[split],
                },
            )
            for split in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        ]

    def _generate_examples(self, data_dir) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        for dirpath, _, filenames in os.walk(data_dir):
            for guid, filename in enumerate(filenames):
                if filename.endswith(".txt"):
                    txt_file_path = Path(dirpath, filename)
                    if self.config.schema == "source":
                        example = parsing.parse_brat_file(txt_file_path, annotation_file_suffixes=[".a1", ".rel"])
                        example["id"] = str(guid)
                        for key in ["events", "attributes", "normalizations"]:
                            del example[key]
                        yield guid, example
                    elif self.config.schema == "bigbio_kb":
                        example = parsing.brat_parse_to_bigbio_kb(
                            parsing.parse_brat_file(txt_file_path), entity_types=_ENTITY_TYPES
                        )
                        example["id"] = str(guid)
                        yield guid, example
