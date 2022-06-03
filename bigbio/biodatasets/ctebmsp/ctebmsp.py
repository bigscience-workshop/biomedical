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
The Clinical Trials for Evidence-Based Medicine in Spanish (CT-EBM-SP) Corpus
gathers 1200 texts about clinical trial studies for NER; this resource contains
500 abstracts of journal articles about clinical trials and 700 announcements
of trial protocols (292 173 tokens), with 46 699 annotated entities.

Entities were annotated according to the Unified Medical Language System (UMLS)
semantic groups: anatomy (ANAT), pharmacological and chemical substances (CHEM),
pathologies (DISO), and lab tests, diagnostic or therapeutic procedures (PROC).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.ES]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{CampillosLlanos2021,
  author    = {Leonardo Campillos-Llanos and
               Ana Valverde-Mateos and
               Adri{\'{a}}n Capllonch-Carri{\'{o}}n and
               Antonio Moreno-Sandoval},
  title     = {A clinical trials corpus annotated with {UMLS}
               entities to enhance the access to evidence-based medicine},
  journal   = {{BMC} Medical Informatics and Decision Making},
  volume    = {21},
  year      = {2021},
  url       = {https://doi.org/10.1186/s12911-021-01395-z},
  doi       = {10.1186/s12911-021-01395-z},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "ctebmsp"

_ABSTRACTS_DESCRIPTION = """\
The "abstracts" subset of the Clinical Trials for Evidence-Based Medicine in Spanish
(CT-EBM-SP) corpus contains 500 abstracts of clinical trial studies in Spanish,
published in journals with a Creative Commons license. Most were downloaded from
the SciELO repository and free abstracts in PubMed.

Abstracts were retrieved with the query:
Clinical Trial[ptyp] AND “loattrfree full text”[sb] AND “spanish”[la].

(Information collected from 10.1186/s12911-021-01395-z)
"""

_EUDRACT_DESCRIPTION = """\
The "abstracts" subset of the Clinical Trials for Evidence-Based Medicine in Spanish
(CT-EBM-SP) corpus contains 500 abstracts of clinical trial studies in Spanish,
published in journals with a Creative Commons license. Most were downloaded from
the SciELO repository and free abstracts in PubMed.

Abstracts were retrieved with the query:
Clinical Trial[ptyp] AND “loattrfree full text”[sb] AND “spanish”[la].

(Information collected from 10.1186/s12911-021-01395-z)
"""

_DESCRIPTION = {
    "ctebmsp_abstracts": _ABSTRACTS_DESCRIPTION,
    "ctebmsp_eudract": _EUDRACT_DESCRIPTION,
}

_HOMEPAGE = "http://www.lllf.uam.es/ESP/nlpmedterm_en.html"

_LICENSE = Licenses.CC_BY_NC_4p0

_URLS = {
    _DATASETNAME: "http://www.lllf.uam.es/ESP/nlpdata/wp2/CT-EBM-SP.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class CTEBMSpDataset(datasets.GeneratorBasedBuilder):
    """A Spanish clinical trials corpus annotated with UMLS entities"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    for study in ["abstracts", "eudract"]:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"ctebmsp_{study}_source",
                version=SOURCE_VERSION,
                description=f"CT-EBM-SP {study.capitalize()} source schema",
                schema="source",
                subset_id=f"ctebmsp_{study}",
            )
        )

        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"ctebmsp_{study}_bigbio_kb",
                version=BIGBIO_VERSION,
                description=f"CT-EBM-SP {study.capitalize()} BigBio schema",
                schema="bigbio_kb",
                subset_id=f"ctebmsp_{study}",
            ),
        )

    DEFAULT_CONFIG_NAME = "ctebmsp_abstracts_source"

    # Entities from the Unified Medical Language System (UMLS) semantic groups

    def _info(self) -> datasets.DatasetInfo:
        """
        Provide information about CT-EBM-SP
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
            description=_DESCRIPTION[self.config.subset_id],
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls))
        studies_path = {
            "ctebmsp_abstracts": "abstracts",
            "ctebmsp_eudract": "eudract",
        }

        study_path = studies_path[self.config.subset_id]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"dir_files": data_dir / "train" / study_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"dir_files": data_dir / "test" / study_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"dir_files": data_dir / "dev" / study_path},
            ),
        ]

    def _generate_examples(self, dir_files) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        txt_files = list(dir_files.glob("*txt"))

        if self.config.schema == "source":
            for guid, txt_file in enumerate(txt_files):
                example = parsing.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example

        elif self.config.schema == "bigbio_kb":
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file)
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
