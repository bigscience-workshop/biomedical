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
from bigbio.utils.constants import Tasks

_DATASETNAME = "bionlp_st_2011_ge"
_SOURCE_VIEW_NAME = "source"
_UNIFIED_VIEW_NAME = "bigbio"

_LOCAL = False
_CITATION = """\
@inproceedings{10.5555/2107691.2107693,
author = {Kim, Jin-Dong and Wang, Yue and Takagi, Toshihisa and Yonezawa, Akinori},
title = {Overview of Genia Event Task in BioNLP Shared Task 2011},
year = {2011},
isbn = {9781937284091},
publisher = {Association for Computational Linguistics},
address = {USA},
abstract = {The Genia event task, a bio-molecular event extraction task,
is arranged as one of the main tasks of BioNLP Shared Task 2011.
As its second time to be arranged for community-wide focused
efforts, it aimed to measure the advance of the community since 2009,
and to evaluate generalization of the technology to full text papers.
After a 3-month system development period, 15 teams submitted their
performance results on test cases. The results show the community has
made a significant advancement in terms of both performance improvement
and generalization.},
booktitle = {Proceedings of the BioNLP Shared Task 2011 Workshop},
pages = {7–15},
numpages = {9},
location = {Portland, Oregon},
series = {BioNLP Shared Task '11}
}
"""

_DESCRIPTION = """\
The BioNLP-ST GE task has been promoting development of fine-grained information extraction (IE) from biomedical
documents, since 2009. Particularly, it has focused on the domain of NFkB as a model domain of Biomedical IE.
The GENIA task aims at extracting events occurring upon genes or gene products, which are typed as "Protein"
without differentiating genes from gene products. Other types of physical entities, e.g. cells, cell components,
are not differentiated from each other, and their type is given as "Entity".
"""

_HOMEPAGE = "https://sites.google.com/site/bionlpst/bionlp-shared-task-2011/genia-event-extraction-genia"

_LICENSE = "https://creativecommons.org/licenses/by/3.0/ CC-BY-3.0"

_URLs = {
    "source": "https://github.com/openbiocorpora/bionlp-st-2011-ge/archive/refs/heads/master.zip",
    "bigbio_kb": "https://github.com/openbiocorpora/bionlp-st-2011-ge/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [
    Tasks.EVENT_EXTRACTION,
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.COREFERENCE_RESOLUTION,
]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class bionlp_st_2011_ge(datasets.GeneratorBasedBuilder):
    """The BioNLP-ST GE task has been promoting development of fine-grained information extraction (IE) from biomedical
    documents, since 2009. Particularly, it has focused on the domain of NFkB as a model domain of Biomedical IE"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bionlp_st_2011_ge_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2011_ge source schema",
            schema="source",
            subset_id="bionlp_st_2011_ge",
        ),
        BigBioConfig(
            name="bionlp_st_2011_ge_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bionlp_st_2011_ge BigBio schema",
            schema="bigbio_kb",
            subset_id="bionlp_st_2011_ge",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bionlp_st_2011_ge_source"

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
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        my_urls = _URLs[self.config.schema]
        data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": data_dir / f"bionlp-st-2011-ge-master" / "original-data" / "train",
            "dev": data_dir / f"bionlp-st-2011-ge-master" / "original-data" / "devel",
            "test": data_dir / f"bionlp-st-2011-ge-master" / "original-data" / "test",
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
                example = parsing.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file)
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
