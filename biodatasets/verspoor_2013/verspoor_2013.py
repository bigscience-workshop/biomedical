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
This dataset contains annotations for a small corpus of full text journal publications
on the subject of inherited colorectal cancer. It is suitable for Named Entity Recognition and
Relation Extraction tasks. It uses the Variome Annotation Schema,  a schema that aims to
capture the core concepts and relations relevant to cataloguing  and interpreting human
genetic variation and its relationship to disease, as described in the published literature.
The schema was inspired by the needs of the database curators of the International Society
for Gastrointestinal Hereditary Tumours (InSiGHT) database, but is intended to have
application to genetic variation information in a range of diseases.
"""

from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LOCAL = False
_CITATION = """\
@article{verspoor2013annotating,
title={Annotating the biomedical literature for the human variome},
author={Verspoor, Karin and Jimeno Yepes, Antonio and Cavedon, Lawrence and McIntosh, Tara and Herten-Crabb, Asha and Thomas, Zo{"e} and Plazzer, John-Paul},
journal={Database},
volume={2013},
year={2013},
publisher={Oxford Academic}
}
"""  # noqa: E501


_DATASETNAME = "verspoor_2013"

_DESCRIPTION = """\
This dataset contains annotations for a small corpus of full text journal publications
on the subject of inherited colorectal cancer. It is suitable for Named Entity Recognition and
Relation Extraction tasks. It uses the Variome Annotation Schema,  a schema that aims to
capture the core concepts and relations relevant to cataloguing  and interpreting human
genetic variation and its relationship to disease, as described in the published literature.
The schema was inspired by the needs of the database curators of the International Society
for Gastrointestinal Hereditary Tumours (InSiGHT) database, but is intended to have
application to genetic variation information in a range of diseases."""


_HOMEPAGE = "NA"

_LICENSE = "NA"

_URLS = ["http://github.com/rockt/SETH/zipball/master/"]

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.RELATION_EXTRACTION,
]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class Verspoor2013Dataset(datasets.GeneratorBasedBuilder):
    """This dataset contains annotations for a small corpus of full text journal publications
    on the subject of inherited colorectal cancer. It is suitable for Named Entity Recognition and
    Relation Extraction tasks. It uses the Variome Annotation Schema,  a schema that aims to
    capture the core concepts and relations relevant to cataloguing  and interpreting human
    genetic variation and its relationship to disease, as described in the published literature.
    The schema was inspired by the needs of the database curators of the International Society
    for Gastrointestinal Hereditary Tumours (InSiGHT) database, but is intended to have
    application to genetic variation information in a range of diseases."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="verspoor_2013_source",
            version=SOURCE_VERSION,
            description="verspoor_2013 source schema",
            schema="source",
            subset_id="verspoor_2013",
        ),
        BigBioConfig(
            name="verspoor_2013_bigbio_kb",
            version=BIGBIO_VERSION,
            description="verspoor_2013 BigBio schema",
            schema="bigbio_kb",
            subset_id="verspoor_2013",
        ),
    ]

    DEFAULT_CONFIG_NAME = "verspoor_2013_source"

    _ENTITY_TYPES = {
        "Concepts_Ideas",
        "Disorder",
        "Phenomena",
        "Physiology",
        "age",
        "body-part",
        "cohort-patient",
        "disease",
        "ethnicity",
        "gender",
        "gene",
        "mutation",
        "size",
    }

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
                    "events": [  # E line in brat
                        {
                            "trigger": datasets.Value("string"),  # refers to the text_bound_annotation of the trigger,
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
                            "resource_name": datasets.Value("string"),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value("string"),  # ID in the resource, e.g. 534366
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

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        # Download gets entire git repo containing unused data from other datasets
        repo_dir = Path(dl_manager.download_and_extract(_URLS[0]))
        data_dir = repo_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Find the relevant files from Verspor2013 and move them to a new directory
        verspoor_files = repo_dir.glob("*/*/*Verspoor2013/**/*")
        for file in verspoor_files:
            if file.is_file() and "readme" not in str(file):
                file.rename(data_dir / file.name)

        # Delete all unused files and directories from the original download
        for x in repo_dir.glob("[!data]*"):
            if x.is_file():
                x.unlink()
            elif x.is_dir():
                rmtree(x)

        data_files = {"text_files": list(data_dir.glob("*.txt"))}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_files": data_files,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, data_files, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            txt_files = data_files["text_files"]
            for guid, txt_file in enumerate(txt_files):
                example = parsing.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example

        elif self.config.schema == "bigbio_kb":
            txt_files = data_files["text_files"]
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file), entity_types=self._ENTITY_TYPES
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
