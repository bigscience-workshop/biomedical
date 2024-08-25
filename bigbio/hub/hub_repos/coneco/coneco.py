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
A dataset loading script for the Complex Named Entity Corpus (CoNECo.)

CoNECo is an annotated corpus for NER and NEN of protein-containing complexes. \
CoNECo comprises 1,621 documents with 2,052 entities, 1,976 of which are normalized \
to Gene Ontology. We divided the corpus into training, development, and test sets.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils.license import Licenses

from .bigbiohub import (BigBioConfig, Tasks, brat_parse_to_bigbio_kb,
                        kb_features, parse_brat_file)

_LOCAL = False
_CITATION = """\
@article {Nastou2024.05.18.594800,
    author = {Nastou, Katerina and Koutrouli, Mikaela and Pyysalo, Sampo and Jensen, Lars Juhl},
    title = {CoNECo: A Corpus for Named Entity recognition and normalization of protein Complexes},
    elocation-id = {2024.05.18.594800},
    year = {2024},
    doi = {10.1101/2024.05.18.594800},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2024/05/29/2024.05.18.594800},
    eprint = {https://www.biorxiv.org/content/early/2024/05/29/2024.05.18.594800.full.pdf},
    journal = {bioRxiv}
}
"""

_DATASETNAME = "coneco"
_DISPLAYNAME = "CoNECo"

_DESCRIPTION = """\
Complex Named Entity Corpus (CoNECo) is an annotated corpus for NER and NEN of protein-containing complexes. \
CoNECo comprises 1,621 documents with 2,052 entities, 1,976 of which are normalized to Gene Ontology. We \
divided the corpus into training, development, and test sets.
"""

_HOMEPAGE = "https://zenodo.org/records/11263147"

_LICENSE = Licenses.CC_BY_4p0

_URLS = {
    _DATASETNAME: "https://zenodo.org/records/11263147/files/CoNECo_corpus.tar.gz?download=1",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
]

_SOURCE_VERSION = "2.0.0"

_BIGBIO_VERSION = "1.0.0"


class ConecoDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="coneco_source",
            version=SOURCE_VERSION,
            description="coneco source schema",
            schema="source",
            subset_id="coneco",
        ),
        BigBioConfig(
            name="coneco_bigbio_kb",
            version=BIGBIO_VERSION,
            description="coneco BigBio schema",
            schema="bigbio_kb",
            subset_id="coneco",
        ),
    ]

    DEFAULT_CONFIG_NAME = "coneco_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [  # T line in brat, i.e. entities for NER task
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "normalizations": [  # N lines in brat, i.e. normalization for NEN task
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value("string"),
                            "cuid": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = kb_features

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
        data_dir = Path(dl_manager.download_and_extract(urls))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir / "train",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir / "test",
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir / "dev",
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            for file in filepath.iterdir():
                if file.suffix != ".txt":
                    continue
                brat_parsed = parse_brat_file(file)
                brat_parsed["id"] = file.stem

                yield brat_parsed["document_id"], brat_parsed

        elif self.config.schema == "bigbio_kb":
            for file in filepath.iterdir():
                if file.suffix != ".txt":
                    continue
                brat_parsed = parse_brat_file(file)
                bigbio_kb_example = brat_parse_to_bigbio_kb(brat_parsed)
                bigbio_kb_example["id"] = file.stem

                yield bigbio_kb_example["id"], bigbio_kb_example
