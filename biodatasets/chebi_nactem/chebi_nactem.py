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

import os
import re
from typing import Dict, List, Set, Tuple

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@InProceedings{Shardlow2018,
author = {Shardlow, M J and Nguyen, N and Owen, G and O'Donovan, C and Leach, A and McNaught, J and Turner,
S and Ananiadou, S},
booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)},
title = {A New Corpus to Support Text Mining for the Curation of Metabolites in the {ChEBI} Database},
year = {2018},
month = may,
pages = {280--285},
conference = {Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
language = {en},
location = {Miyazaki, Japan},
}
"""

_DATASETNAME = "chebi_nactem"

_DESCRIPTION = """\
The ChEBI corpus contains 199 annotated abstracts and 100 annotated full papers.
All documents in the corpus have been annotated for named entities and relations between these.
In total, our corpus provides over 15000 named entity annotations and over 6,000 relations between entities.
"""

_HOMEPAGE = "http://www.nactem.ac.uk/chebi"

_LICENSE = "Creative Commons Attribution 4.0 International License. (https://creativecommons.org/licenses/by/4.0/)"

_URLS = {
    _DATASETNAME: "http://www.nactem.ac.uk/chebi/ChEBI.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class ChebiNactemDatasset(datasets.GeneratorBasedBuilder):
    """Chemical Entities of Biological Interest (ChEBI) corpus."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []
    for subset_id in ["abstr_ann1", "abstr_ann2", "fullpaper"]:
        BUILDER_CONFIGS += [
            BigBioConfig(
                name=f"chebi_nactem_{subset_id}_source",
                version=SOURCE_VERSION,
                description="chebi_nactem source schema",
                schema="source",
                subset_id=f"chebi_nactem_{subset_id}",
            ),
            BigBioConfig(
                name=f"chebi_nactem_{subset_id}_bigbio_kb",
                version=BIGBIO_VERSION,
                description="chebi_nactem BigBio schema",
                schema="bigbio_kb",
                subset_id=f"chebi_nactem_{subset_id}",
            ),
        ]

    DEFAULT_CONFIG_NAME = "chebi_nactem_fullpaper_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1": datasets.Value("string"),
                            "arg2": datasets.Value("string"),
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
        else:
            raise NotImplementedError(self.config.schema)

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

        subset_paths = {
            "chebi_nactem_abstr_ann1": "ChEBI/abstracts/Annotator1",
            "chebi_nactem_abstr_ann2": "ChEBI/abstracts/Annotator2",
            "chebi_nactem_fullpaper": "ChEBI/fullpapers",
        }

        subset_dir = os.path.join(data_dir, subset_paths[self.config.subset_id])

        doc_ids = set()
        for file_name in os.listdir(subset_dir):
            _id = file_name.split(".")[0]
            doc_ids.add(_id)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": subset_dir,
                    "document_ids": doc_ids,
                },
            )
        ]

    def _generate_examples(self, data_dir: str, document_ids: Set[str]) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        label_pos_pattern = re.compile(r"(?P<entity>\S+) (?P<offsets>(;?\d+ \d+)+)")

        def uid_gen():
            _uid = 0
            while True:
                yield str(_uid)
                _uid += 1

        uid = iter(uid_gen())

        for idx, doc_id in enumerate(document_ids):
            doc_file = os.path.join(data_dir, doc_id + ".txt")
            with open(doc_file) as handle:
                text = handle.read()

            ann_file = os.path.join(data_dir, doc_id + ".ann")

            annotations = []
            relations = []

            with open(ann_file) as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith("#"):
                        continue
                    if line.startswith("T"):
                        ann_id, label_pos, ann_text = line.split("\t")
                        match = label_pos_pattern.fullmatch(label_pos)
                        entity = match.group("entity")
                        offsets = [p.split() for p in match.group("offsets").split(";")]
                        offsets = [(int(p[0]), int(p[1])) for p in offsets]

                        annotations.append((ann_id, entity, offsets, ann_text))
                    else:
                        assert line.startswith("R")
                        rel_id, label_spans = line.split("\t")
                        relation, arg1, arg2 = label_spans.split()
                        arg1 = arg1.split(":")[1]
                        arg2 = arg2.split(":")[1]

                        relations.append((rel_id, relation, arg1, arg2))

            if self.config.schema == "source":
                yield idx, {
                    "document_id": doc_id,
                    "text": text,
                    "entities": [
                        {
                            "id": ann_id,
                            "type": entity,
                            "text": ann_text,
                            "offsets": offsets,
                        }
                        for ann_id, entity, offsets, ann_text in annotations
                    ],
                    "relations": [
                        {
                            "id": rel_id,
                            "type": relation,
                            "arg1": arg1,
                            "arg2": arg2,
                        }
                        for rel_id, relation, arg1, arg2 in relations
                    ],
                }
            elif self.config.schema == "bigbio_kb":
                yield idx, {
                    "id": next(uid),
                    "document_id": doc_id,
                    "passages": [
                        {
                            "id": next(uid),
                            "type": "",
                            "text": [text],
                            "offsets": [(0, len(text))],
                        }
                    ],
                    "entities": [
                        {
                            "id": f"{idx}_{ann_id}",
                            "type": entity,
                            "text": [ann_text],
                            "offsets": offsets,
                            "normalized": [],
                        }
                        for ann_id, entity, offsets, ann_text in annotations
                    ],
                    "events": [],
                    "coreferences": [],
                    "relations": [
                        {
                            "id": f"{idx}_{rel_id}",
                            "type": relation,
                            "arg1_id": f"{idx}_{arg1}",
                            "arg2_id": f"{idx}_{arg2}",
                            "normalized": [],
                        }
                        for rel_id, relation, arg1, arg2 in relations
                    ],
                }
            else:
                raise NotImplementedError(self.config.schema)
