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
from typing import Dict, List

import datasets

from .bigbiohub import kb_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_DATASETNAME = "bionlp_st_2013_cg"
_DISPLAYNAME = "BioNLP 2013 CG"

_UNIFIED_VIEW_NAME = "bigbio"

_LANGUAGES = ['English']
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{pyysalo-etal-2013-overview,
    title = "Overview of the Cancer Genetics ({CG}) task of {B}io{NLP} Shared Task 2013",
    author = "Pyysalo, Sampo  and
      Ohta, Tomoko  and
      Ananiadou, Sophia",
    booktitle = "Proceedings of the {B}io{NLP} Shared Task 2013 Workshop",
    month = aug,
    year = "2013",
    address = "Sofia, Bulgaria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W13-2008",
    pages = "58--66",
}
"""

_DESCRIPTION = """\
the Cancer Genetics (CG) is a event extraction task and a main task of the BioNLP Shared Task (ST) 2013.
The CG task is an information extraction task targeting the recognition of events in text,
represented as structured n-ary associations of given physical entities. In addition to
addressing the cancer domain, the CG task is differentiated from previous event extraction
tasks in the BioNLP ST series in addressing a wide range of pathological processes and multiple
levels of biological organization, ranging from the molecular through the cellular and organ
levels up to whole organisms. Final test set submissions were accepted from six teams
"""

_HOMEPAGE = "https://github.com/openbiocorpora/bionlp-st-2013-cg"

_LICENSE = 'GENIA Project License for Annotated Corpora'

_URLs = {
    "bionlp_st_2013_cg": "https://github.com/openbiocorpora/bionlp-st-2013-cg/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [
    Tasks.EVENT_EXTRACTION,
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.COREFERENCE_RESOLUTION,
]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class bionlp_st_2013_cg(datasets.GeneratorBasedBuilder):
    """the Cancer Genetics (CG) is a event extraction task
    and a main task of the BioNLP Shared Task (ST) 2013."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bionlp_st_2013_cg_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2013 source schema",
            schema="source",
            subset_id="bionlp_st_2013_pc",
        ),
        BigBioConfig(
            name="bionlp_st_2013_cg_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bionlp_st_2013_cg BigBio schema",
            schema="bigbio_kb",
            subset_id="bionlp_st_2013_pc",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bionlp_st_2013_cg_source"

    _ROLE_MAPPING = {
        "Theme2": "Theme",
        "Theme3": "Theme",
        "Theme4": "Theme",
        "Theme5": "Theme",
        "Theme6": "Theme",
        "Instrument2": "Instrument",
        "Instrument3": "Instrument",
        "Participant2": "Participant",
        "Participant3": "Participant",
        "Participant4": "Participant",
        "Cause2": "Cause",
    }

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
            features = kb_features
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # This is not applicable for MLEE.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=str(_LICENSE),
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        my_urls = _URLs[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": data_dir / f"bionlp-st-2013-cg-master" / "original-data" / "train",
            "dev": data_dir / f"bionlp-st-2013-cg-master" / "original-data" / "devel",
            "test": data_dir / f"bionlp-st-2013-cg-master" / "original-data" / "test",
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

    def _standardize_arguments_roles(self, kb_example: Dict) -> Dict:

        for event in kb_example["events"]:
            for argument in event["arguments"]:
                role = argument["role"]
                argument["role"] = self._ROLE_MAPPING.get(role, role)

        return kb_example

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
                example = self._standardize_arguments_roles(example)
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
