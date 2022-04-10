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
A dataset loading script for the PharmaCoNER corpus.

The PharmaCoNER datset is a manually annotated collection of clinical case
studies derived from the Spanish Clinical Case Corpus (SPACCC). It was designed
for the Pharmacological Substances, Compounds and Proteins NER track, the first
shared task on detecting drug and chemical entities in Spanish medical documents.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from utils import parsing, schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@inproceedings{,
    title = "PharmaCoNER: Pharmacological Substances, Compounds and proteins Named Entity Recognition track",
    author = "Gonzalez-Agirre, Aitor  and
      Marimon, Montserrat  and
      Intxaurrondo, Ander  and
      Rabal, Obdulia  and
      Villegas, Marta  and
      Krallinger, Martin",
    booktitle = "Proceedings of The 5th Workshop on BioNLP Open Shared Tasks",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-5701",
    doi = "10.18653/v1/D19-5701",
    pages = "1--10",
}
"""

_DATASETNAME = "pharmaconer"

_DESCRIPTION = """\
PharmaCoNER: Pharmacological Substances, Compounds and Proteins Named Entity Recognition track

This dataset is designed for the PharmaCoNER task, sponsored by Plan de Impulso de las Tecnologías del Lenguaje.

It is a manually classified collection of clinical case studies derived from the
Spanish Clinical Case Corpus (SPACCC), an open access electronic library that gathers
Spanish medical publications from SciELO (Scientific Electronic Library Online).

The annotation of the entire set of entity mentions was carried out by medicinal chemistry experts
and it includes the following 4 entity types: NORMALIZABLES, NO_NORMALIZABLES, PROTEINAS and UNCLEAR.

The PharmaCoNER corpus contains a total of 396,988 words and 1,000 clinical cases that
have been randomly sampled into 3 subsets: train (500), validation (250) and test (250).

For further information, please visit https://temu.bsc.es/pharmaconer/ or send an email to encargo-pln-life@bsc.es
"""

_HOMEPAGE = "https://temu.bsc.es/pharmaconer/index.php/datasets/"

_LICENSE = "Creative Commons Attribution 4.0 International"

_URLS = {
    "pharmaconer": "https://zenodo.org/record/4270158/files/pharmaconer.zip?download=1",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.1.0"

_BIGBIO_VERSION = "1.0.0"


class PharmaconerDataset(datasets.GeneratorBasedBuilder):
    """Manually annotated collection of clinical case studies from Spanish medical publications."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pharmaconer_source",
            version=SOURCE_VERSION,
            description="PharmaCoNER source schema",
            schema="source",
            subset_id="pharmaconer",
        ),
        BigBioConfig(
            name="pharmaconer_bigbio_kb",
            version=BIGBIO_VERSION,
            description="PharmaCoNER BigBio schema",
            schema="bigbio_kb",
            subset_id="pharmaconer",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pharmaconer_source"

    _ENTITY_TYPES = {"NORMALIZABLES", "PROTEINAS", "NO_NORMALIZABLES", "UNCLEAR"}

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
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
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """
        Downloads/extracts the data to generate the train, validation and test splits.

        Each split is created by instantiating a `datasets.SplitGenerator`, which will
        call `this._generate_examples` with the keyword arguments in `gen_kwargs`.
        """

        data_dir = dl_manager.download_and_extract(_URLS["pharmaconer"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "pharmaconer/train-set_1.1/train/subtrack1")),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "pharmaconer/test-set_1.1/test/subtrack1")),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "pharmaconer/dev-set_1.1/dev/subtrack1")),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """
        This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        Method parameters are unpacked from `gen_kwargs` as given in `_split_generators`.
        """

        txt_files = list(filepath.glob("*txt"))

        if self.config.schema == "source":
            for guid, txt_file in enumerate(txt_files):
                example = parsing.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file), entity_types=self._ENTITY_TYPES
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
