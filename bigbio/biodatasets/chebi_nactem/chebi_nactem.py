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
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses
from bigbio.utils.parsing import parse_brat_file

_TAGS = []
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
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

_LICENSE = Licenses.CC_BY_4p0

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
            license=str(_LICENSE),
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

        subset_dir = Path(data_dir) / subset_paths[self.config.subset_id]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": subset_dir,
                },
            )
        ]

    def _generate_examples(self, data_dir: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        def uid_gen():
            _uid = 0
            while True:
                yield str(_uid)
                _uid += 1

        uid = iter(uid_gen())

        txt_files = (f for f in os.listdir(data_dir) if f.endswith(".txt"))
        for idx, file_name in enumerate(txt_files):

            brat_file = data_dir / file_name
            contents = parse_brat_file(brat_file)

            if self.config.schema == "source":
                yield idx, {
                    "document_id": contents["document_id"],
                    "text": contents["text"],
                    "entities": contents["text_bound_annotations"],
                    "relations": [
                        {
                            "id": relation["id"],
                            "type": relation["type"],
                            "arg1": relation["head"]["ref_id"],
                            "arg2": relation["tail"]["ref_id"],
                        }
                        for relation in contents["relations"]
                    ],
                }

            elif self.config.schema == "bigbio_kb":
                yield idx, {
                    "id": next(uid),
                    "document_id": contents["document_id"],
                    "passages": [
                        {
                            "id": next(uid),
                            "type": "",
                            "text": [contents["text"]],
                            "offsets": [(0, len(contents["text"]))],
                        }
                    ],
                    "entities": [
                        {
                            "id": f"{idx}_{entity['id']}",
                            "type": entity["type"],
                            "offsets": entity["offsets"],
                            "text": entity["text"],
                            "normalized": [],
                        }
                        for entity in contents["text_bound_annotations"]
                    ],
                    "events": [],
                    "coreferences": [],
                    "relations": [
                        {
                            "id": f"{idx}_{relation['id']}",
                            "type": relation["type"],
                            "arg1_id": f"{idx}_{relation['head']['ref_id']}",
                            "arg2_id": f"{idx}_{relation['tail']['ref_id']}",
                            "normalized": [],
                        }
                        for relation in contents["relations"]
                    ],
                }
            else:
                raise NotImplementedError(self.config.schema)
