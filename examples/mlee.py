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

"""
MLEE is an event extraction corpus consisting of manually annotated abstracts of papers
on angiogenesis. It contains annotations for entities, relations, events and coreferences
The annotations span molecular, cellular, tissue, and organ-level processes.
"""
from pathlib import Path
from typing import Dict, List

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_DATASETNAME = "mlee"
_SOURCE_VIEW_NAME = "source"
_UNIFIED_VIEW_NAME = "bigbio"

_LANGUAGES=[Lang.EN]
_LOCAL = False
_CITATION = """\
@article{,
  author = {Pyysalo, Sampo and Ohta, Tomoko and Miwa, Makoto and Cho, Han-Cheol and Tsujii, Jun'ichi and Ananiadou, Sophia},
  title = "{Event extraction across multiple levels of biological organization}",
  journal   = {Bioinformatics},
  volume    = {28},
  year      = {2012},
  url       = {https://doi.org/10.1093/bioinformatics/bts407},
  doi       = {10.1093/bioinformatics/bts407},
  biburl    = {},
  bibsource = {}
}
"""

_DESCRIPTION = """\
MLEE is an event extraction corpus consisting of manually annotated abstracts of papers
on angiogenesis. It contains annotations for entities, relations, events and coreferences
The annotations span molecular, cellular, tissue, and organ-level processes.
"""

_HOMEPAGE = "http://www.nactem.ac.uk/MLEE/"

_LICENSE = "CC BY-NC-SA 3.0"

_URLs = {
    "source": "http://www.nactem.ac.uk/MLEE/MLEE-1.0.2-rev1.tar.gz",
    "bigbio_kb": "http://www.nactem.ac.uk/MLEE/MLEE-1.0.2-rev1.tar.gz",
}

_SUPPORTED_TASKS = [
    Tasks.EVENT_EXTRACTION,
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.RELATION_EXTRACTION,
    Tasks.COREFERENCE_RESOLUTION,
]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class MLEE(datasets.GeneratorBasedBuilder):
    """Write a short docstring documenting what this dataset is"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="mlee_source",
            version=SOURCE_VERSION,
            description="MLEE source schema",
            schema="source",
            subset_id="mlee",
        ),
        BigBioConfig(
            name="mlee_bigbio_kb",
            version=SOURCE_VERSION,
            description="MLEE BigBio schema",
            schema="bigbio_kb",
            subset_id="mlee",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mlee_source"

    _ROLE_MAPPING = {
        "Theme2": "Theme",
        "Instrument2": "Instrument",
        "Participant2": "Participant",
        "Participant3": "Participant",
        "Participant4": "Participant",
    }

    def _info(self):
        """
        Provide information about MLEE:
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
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """
        Create the three splits provided by MLEE: train, validation and test.

        Each split is created by instantiating a `datasets.SplitGenerator`, which will
        call `this._generate_examples` with the keyword arguments in `gen_kwargs`.
        """

        my_urls = _URLs[self.config.schema]
        data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": data_dir
            / "MLEE-1.0.2-rev1"
            / "standoff"
            / "development"
            / "train",
            "dev": data_dir / "MLEE-1.0.2-rev1" / "standoff" / "development" / "test",
            "test": data_dir / "MLEE-1.0.2-rev1" / "standoff" / "test" / "test",
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
        """
        Yield one `(guid, example)` pair per abstract in MLEE.
        The contents of `example` will depend on the chosen configuration.
        """
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
