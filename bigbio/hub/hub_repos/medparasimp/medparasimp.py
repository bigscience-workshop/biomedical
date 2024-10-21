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
Paragraph-level Simplification of Medical Texts ("MedParaSimp") is a
dataset that contains pairs of technical medical abstracts from
biomedical systematic reviews (taken from the Cochrane Library)
and their corresponding plain-language summarizations (PLS).
The PLS's were created by the authors of the original abstracts.
The dataset was obtained by scraping the Cochrane Library website.
"""

from typing import Dict, List, Tuple

import datasets

from .bigbiohub import BigBioConfig, Tasks, text2text_features

_LOCAL = False

_CITATION = """\
@inproceedings{devaraj-etal-2021-paragraph,
    title = "Paragraph-level Simplification of Medical Texts",
    author = "Devaraj, Ashwin and Marshall, Iain and Wallace, Byron and Li, Junyi Jessy",
    booktitle = {Proceedings of the 2021 Conference of the North
                American Chapter of the Association for Computational Linguistics},
    month = jun,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.395",
    pages = "4972--4984",
}
"""

_DATASETNAME = "medparasimp"

_DESCRIPTION = """\
This dataset is designed for the summarization NLP task. It is a
collection of technical abstracts of biomedical systematic reviews
and corresponding plain-language summaries (PLS) from the Cochrane
Database of Systematic Reviews, which comprises thousands of evidence
synopses (where authors provide an overview of all published evidence
relevant to a particular clinical question or topic). The PLS are
written by review authors; Cochrane’s PLS standards recommend that
“the PLS should be written in plain English which can be understood by
most readers without a university education”. PLS are not parallel with
every sentence in the abstract; on the contrary, they are structured heterogeneously.
"""

_HOMEPAGE = "https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts"

_LICENSE = "CC_BY_4p0"

_URLS = {
    _DATASETNAME: {
        "train_doi": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/train.doi"
        ),
        "train_source": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/train.source"
        ),
        "train_target": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/train.target"
        ),
        "val_doi": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/val.doi"
        ),
        "val_source": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/val.source"
        ),
        "val_target": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/val.target"
        ),
        "test_doi": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/test.doi"
        ),
        "test_source": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/test.source"
        ),
        "test_target": (
            "https://raw.githubusercontent.com/AshOlogn/"
            "Paragraph-level-Simplification-of-Medical-Texts/main/data/data-1024/test.target"
        ),
    }
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

_LANGUAGES = ["English (United States)"]

_PUBMED = False

_DISPLAYNAME = "Paragraph-Level Simplification of Medical Texts"


class MedParaSimpDataset(datasets.GeneratorBasedBuilder):
    """Paired abstracts and plain-language summaries from the Cochrane Database of Systematic Reviews."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="medparasimp_source",
            version=SOURCE_VERSION,
            description=(
                "Paragraph-level Simplification of Medical Texts (MedParaSimp) is a"
                "paired dataset of technical medical abstracts and their plain-language summarizations."
            ),
            schema="source",
            subset_id="medparasimp",
        ),
        BigBioConfig(
            name="medparasimp_bigbio_t2t",
            version=BIGBIO_VERSION,
            description=(
                "Paragraph-level Simplification of Medical Texts (MedParaSimp) is a"
                "paired dataset of technical medical abstracts and their plain-language summarizations."
            ),
            schema="bigbio_t2t",
            subset_id="medparasimp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "medparasimp_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "text_1_name": datasets.Value("string"),
                    "text_2_name": datasets.Value("string"),
                }
            )
        elif self.config.schema == "bigbio_t2t":
            features = text2text_features
        else:
            raise ValueError(
                f"Invalid config.schema specified ({self.config.schema}) - must be one of (source|bigbio_t2t)"
            )

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
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "doi_filepath": data_dir["train_doi"],
                    "source_filepath": data_dir["train_source"],
                    "target_filepath": data_dir["train_target"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "doi_filepath": data_dir["val_doi"],
                    "source_filepath": data_dir["val_source"],
                    "target_filepath": data_dir["val_target"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "doi_filepath": data_dir["test_doi"],
                    "source_filepath": data_dir["test_source"],
                    "target_filepath": data_dir["test_target"],
                },
            ),
        ]

    def _generate_examples(self, doi_filepath: str, source_filepath: str, target_filepath: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        # Read data from files
        with open(doi_filepath, "r") as f:
            dois: List[str] = f.read().splitlines()
        with open(source_filepath, "r") as f:
            sources: List[str] = f.read().splitlines()
        with open(target_filepath, "r") as f:
            targets: List[str] = f.read().splitlines()

        for idx, (source, target) in enumerate(zip(sources, targets)):
            key: int = idx
            example: Dict = {
                "id": str(idx),
                "document_id": dois[idx],
                "text_1": source,
                "text_2": target,
                "text_1_name": "abstract",
                "text_2_name": "pls",
            }
            yield (key, example)


if __name__ == "__main__":
    datasets.load_dataset(__file__)
