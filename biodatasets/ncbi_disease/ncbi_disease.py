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
The NCBI disease corpus is fully annotated at the mention and concept level to serve as a research
resource for the biomedical natural language processing community.
"""

import os
from typing import Dict, Iterator, List, Tuple

import datasets
from bioc import pubtator

from biomed_datasets.utils import schemas
from biomed_datasets.utils.configs import BigBioConfig
from biomed_datasets.utils.constants import Tasks

_CITATION = """\
@article{Dogan2014NCBIDC,
    title        = {NCBI disease corpus: A resource for disease name recognition and concept normalization},
    author       = {Rezarta Islamaj Dogan and Robert Leaman and Zhiyong Lu},
    year         = 2014,
    journal      = {Journal of biomedical informatics},
    volume       = 47,
    pages        = {1--10}
}
"""

_DATASETNAME = "ncbi_disease"

_DESCRIPTION = """\
The NCBI disease corpus is fully annotated at the mention and concept level to serve as a research
resource for the biomedical natural language processing community.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/"
_LICENSE = "Public Domain (CC0)"


_URLS = {
    _DATASETNAME: {
        datasets.Split.TRAIN: "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip",
        datasets.Split.TEST: "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip",
        datasets.Split.VALIDATION: "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip",
    }
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class NCBIDiseaseDataset(datasets.GeneratorBasedBuilder):
    """NCBI Disease"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="ncbi_disease_source",
            version=SOURCE_VERSION,
            description="NCBI Disease source schema",
            schema="source",
            subset_id="ncbi_disease",
        ),
        BigBioConfig(
            name="ncbi_disease_bigbio_kb",
            version=BIGBIO_VERSION,
            description="NCBI Disease BigBio schema",
            schema="bigbio_kb",
            subset_id="ncbi_disease",
        ),
    ]

    DEFAULT_CONFIG_NAME = "ncbi_disease_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "mentions": [
                        {
                            "concept_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": datasets.Sequence(datasets.Value("int32")),
                        }
                    ],
                }
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
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        train_filename = "NCBItrainset_corpus.txt"
        test_filename = "NCBItestset_corpus.txt"
        dev_filename = "NCBIdevelopset_corpus.txt"

        train_filepath = os.path.join(data_dir[datasets.Split.TRAIN], train_filename)
        test_filepath = os.path.join(data_dir[datasets.Split.TEST], test_filename)
        dev_filepath = os.path.join(data_dir[datasets.Split.VALIDATION], dev_filename)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_filepath,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_filepath,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev_filepath,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: str, split: str) -> Iterator[Tuple[str, Dict]]:
        if self.config.schema == "source":
            for i, source_example in enumerate(self._pubtator_to_source(filepath)):
                # Some examples are duplicated in NCBI Disease. We have to make them unique to
                # avoid and error from datasets.
                yield str(i) + "_" + source_example["pmid"], source_example

        elif self.config.schema == "bigbio_kb":
            seen = []
            for kb_example in self._pubtator_to_bigbio_kb(filepath):
                # Some examples are duplicated in NCBI Disease. Avoid yielding more than once.
                if kb_example["id"] in seen:
                    continue
                yield kb_example["id"], kb_example
                seen.append(kb_example["id"])

    @staticmethod
    def _pubtator_to_source(filepath: Dict) -> Iterator[Dict]:
        with open(filepath, "r") as f:
            for doc in pubtator.iterparse(f):
                source_example = {
                    "pmid": doc.pmid,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "mentions": [
                        {
                            "concept_id": mention.id,
                            "type": mention.type,
                            "text": mention.text,
                            "offsets": [mention.start, mention.end],
                        }
                        for mention in doc.annotations
                    ],
                }
                yield source_example

    @staticmethod
    def _pubtator_to_bigbio_kb(filepath: Dict) -> Iterator[Dict]:
        with open(filepath, "r") as f:
            unified_example = {}
            for doc in pubtator.iterparse(f):
                unified_example["id"] = doc.pmid
                unified_example["document_id"] = doc.pmid

                unified_example["passages"] = [
                    {
                        "id": doc.pmid + "_title",
                        "type": "title",
                        "text": [doc.title],
                        "offsets": [[0, len(doc.title)]],
                    },
                    {
                        "id": doc.pmid + "_abstract",
                        "type": "abstract",
                        "text": [doc.abstract],
                        "offsets": [
                            [
                                # +1 assumes the title and abstract will be joined by a space.
                                len(doc.title) + 1,
                                len(doc.title) + 1 + len(doc.abstract),
                            ]
                        ],
                    },
                ]

                unified_entities = []
                for i, entity in enumerate(doc.annotations):
                    # We need a unique identifier for this entity, so build it from the document id and entity id
                    unified_entity_id = "_".join([doc.pmid, entity.id, str(i)])
                    # The user can provide a callable that returns the database name.
                    db_name = "omim" if "OMIM" in entity.id else "mesh"
                    unified_entities.append(
                        {
                            "id": unified_entity_id,
                            "type": entity.type,
                            "text": [entity.text],
                            "offsets": [[entity.start, entity.end]],
                            "normalized": [{"db_name": db_name, "db_id": entity.id}],
                        }
                    )

                unified_example["entities"] = unified_entities
                unified_example["relations"] = []
                unified_example["events"] = []
                unified_example["coreferences"] = []

                yield unified_example
