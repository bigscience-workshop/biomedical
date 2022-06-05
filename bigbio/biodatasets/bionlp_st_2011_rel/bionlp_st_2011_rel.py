# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

from pathlib import Path
from typing import List

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_DATASETNAME = "bionlp_st_2011_rel"
_SOURCE_VIEW_NAME = "source"
_UNIFIED_VIEW_NAME = "bigbio"

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{10.5555/2107691.2107703,
author = {Pyysalo, Sampo and Ohta, Tomoko and Tsujii, Jun'ichi},
title = {Overview of the Entity Relations (REL) Supporting Task of BioNLP Shared Task 2011},
year = {2011},
isbn = {9781937284091},
publisher = {Association for Computational Linguistics},
address = {USA},
abstract = {This paper presents the Entity Relations (REL) task,
a supporting task of the BioNLP Shared Task 2011. The task concerns
the extraction of two types of part-of relations between a gene/protein
and an associated entity. Four teams submitted final results for
the REL task, with the highest-performing system achieving 57.7%
F-score. While experiments suggest use of the data can help improve
event extraction performance, the task data has so far received only
limited use in support of event extraction. The REL task continues
as an open challenge, with all resources available from the shared
task website.},
booktitle = {Proceedings of the BioNLP Shared Task 2011 Workshop},
pages = {83–88},
numpages = {6},
location = {Portland, Oregon},
series = {BioNLP Shared Task '11}
}
"""

_DESCRIPTION = """\
The Entity Relations (REL) task is a supporting task of the BioNLP Shared Task 2011.
The task concerns the extraction of two types of part-of relations between a
gene/protein and an associated entity.
"""

_HOMEPAGE = "https://github.com/openbiocorpora/bionlp-st-2011-rel"

_LICENSE = Licenses.GENIA_PROJECT_LICENSE

_URLs = {
    "source": "https://github.com/openbiocorpora/bionlp-st-2011-rel/archive/refs/heads/master.zip",
    "bigbio_kb": "https://github.com/openbiocorpora/bionlp-st-2011-rel/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.RELATION_EXTRACTION,
    Tasks.COREFERENCE_RESOLUTION,
]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class bionlp_st_2011_rel(datasets.GeneratorBasedBuilder):
    """The Entity Relations (REL) task is a supporting task of the BioNLP Shared Task 2011."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bionlp_st_2011_rel_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2011_rel source schema",
            schema="source",
            subset_id="bionlp_st_2011_rel",
        ),
        BigBioConfig(
            name="bionlp_st_2011_rel_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bionlp_st_2011_rel BigBio schema",
            schema="bigbio_kb",
            subset_id="bionlp_st_2011_rel",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bionlp_st_2011_rel_source"

    _FILE_SUFFIX = [".a1", ".rel", ".ann"]

    def _info(self):
        """
        - `features` defines the schema of the parsed data set. The schema depends on the
        chosen `config`: If it is `_SOURCE_VIEW_NAME` the schema is the schema of the
        original data. If `config` is `_UNIFIED_VIEW_NAME`, then the schema is the
        canonical KB-task schema defined in `biomedical/schemas/kb.py`.
        """
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
                    "events": [  # E line in brat
                        {
                            "trigger": datasets.Value(
                                "string"
                            ),  # refers to the text_bound_annotation of the trigger,
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arguments": datasets.Sequence(
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ),
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
                    "attributes": [  # M or A lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                    "normalizations": [  # N lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value(
                                "string"
                            ),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value(
                                "string"
                            ),  # ID in the resource, e.g. 534366
                            "text": datasets.Value(
                                "string"
                            ),  # Human readable description/name of the entity, e.g. "Barack Obama"
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
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        my_urls = _URLs[self.config.schema]
        data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": data_dir
            / f"bionlp-st-2011-rel-master"
            / "original-data"
            / "train",
            "dev": data_dir / f"bionlp-st-2011-rel-master" / "original-data" / "devel",
            "test": data_dir / f"bionlp-st-2011-rel-master" / "original-data" / "test",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_files": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_files": data_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_files": data_files["test"]},
            ),
        ]

    def _generate_examples(self, data_files: Path):
        if self.config.schema == "source":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = parsing.parse_brat_file(txt_file, self._FILE_SUFFIX)
                example["id"] = str(guid)
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file, self._FILE_SUFFIX)
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
