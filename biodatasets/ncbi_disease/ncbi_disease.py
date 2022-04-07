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

from utils import parsing, schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

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
            pubtator_parsed = parsing.parse_pubtator_file(filepath)
            for pubtator_example in pubtator_parsed:
                yield pubtator_example["pmid"], pubtator_example

        elif self.config.schema == "bigbio_kb":
            pubtator_parsed = parsing.parse_pubtator_file(filepath)
            for pubtator_example in pubtator_parsed:
                kb_example = parsing.pubtator_parse_to_bigbio_kb(pubtator_example, get_db_name=self._get_db_name)
                yield kb_example["id"], kb_example

    @staticmethod
    def _get_db_name(mention: Dict) -> str:
        return "omim" if "OMIM" in mention["concept_id"] else "mesh"


# This allows you to run your dataloader with `python ncbi_disease.py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
