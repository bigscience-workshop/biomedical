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
A dataset loading script for the MEDDOCAN corpus.
The MEDDOCAN datset is a manually annotated collection of clinical case
reports derived from the Spanish Clinical Case Corpus (SPACCC). It was designed
for the Medical Document Anonymization Track, the first the first community
challenge task specifically devoted to the anonymization of medical documents in Spanish
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.ANONYMIZATION]
_LANGUAGES = [Lang.ES]
_PUBMED = False
_LOCAL = False
_CITATION = """\
@inproceedings{marimon2019automatic,
  title={Automatic De-identification of Medical Texts in Spanish: the MEDDOCAN Track, Corpus, Guidelines, Methods and Evaluation of Results.},
  author={Marimon, Montserrat and Gonzalez-Agirre, Aitor and Intxaurrondo, Ander and Rodriguez, Heidy and Martin, Jose Lopez and Villegas, Marta and Krallinger, Martin},
  booktitle={IberLEF@ SEPLN},
  pages={618--638},
  year={2019}
}
"""

_DATASETNAME = "meddocan"

_DESCRIPTION = """\
MEDDOCAN: Medical Document Anonymization Track

This dataset is designed for the MEDDOCAN task, sponsored by Plan de Impulso de las Tecnologías del Lenguaje.

It is a manually classified collection of 1,000 clinical case reports derived from the \
Spanish Clinical Case Corpus (SPACCC), enriched with PHI expressions.

The annotation of the entire set of entity mentions was carried out by experts annotators\
and it includes 29 entity types relevant for the annonymiation of medical documents.\
22 of these annotation types are actually present in the corpus: TERRITORIO, FECHAS, \
EDAD_SUJETO_ASISTENCIA, NOMBRE_SUJETO_ASISTENCIA, NOMBRE_PERSONAL_SANITARIO, \
SEXO_SUJETO_ASISTENCIA, CALLE, PAIS, ID_SUJETO_ASISTENCIA, CORREO, ID_TITULACION_PERSONAL_SANITARIO,\
ID_ASEGURAMIENTO, HOSPITAL, FAMILIARES_SUJETO_ASISTENCIA, INSTITUCION, ID_CONTACTO ASISTENCIAL,\
NUMERO_TELEFONO, PROFESION, NUMERO_FAX, OTROS_SUJETO_ASISTENCIA, CENTRO_SALUD, ID_EMPLEO_PERSONAL_SANITARIO
    
For further information, please visit https://temu.bsc.es/meddocan/ or send an email to encargo-pln-life@bsc.es
"""


_HOMEPAGE = "https://temu.bsc.es/meddocan/"

_LICENSE = Licenses.CC_BY_4p0

_URLS = {
    "meddocan": "https://zenodo.org/record/4279323/files/meddocan.zip?download=1",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MeddocanDataset(datasets.GeneratorBasedBuilder):
    """Manually annotated collection of clinical case studies from Spanish medical publications."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="meddocan_source",
            version=SOURCE_VERSION,
            description="Meddocan source schema",
            schema="source",
            subset_id="meddocan",
        ),
        BigBioConfig(
            name="meddocan_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Meddocan BigBio schema",
            schema="bigbio_kb",
            subset_id="meddocan",
        ),
    ]

    DEFAULT_CONFIG_NAME = "meddocan_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    # "labels": [datasets.Value("string")],
                    "text_bound_annotations": [  # T line in brat
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "events": [  # E line in brat
                        {
                            "trigger": datasets.Value("string"),
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
                            "resource_name": datasets.Value("string"),
                            "cuid": datasets.Value("string"),
                            "text": datasets.Value("string"),
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

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """
        Downloads/extracts the data to generate the train, validation and test splits.
        Each split is created by instantiating a `datasets.SplitGenerator`, which will
        call `this._generate_examples` with the keyword arguments in `gen_kwargs`.
        """

        data_dir = dl_manager.download_and_extract(_URLS["meddocan"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "meddocan/train/brat")),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "meddocan/test/brat")),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "meddocan/dev/brat")),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """
        This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        Method parameters are unpacked from `gen_kwargs` as given in `_split_generators`.
        """

        txt_files = sorted(list(filepath.glob("*txt")))
        # tsv_files = sorted(list(filepaths[1].glob("*tsv")))

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
