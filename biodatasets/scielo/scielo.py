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
Parallel corpus of full-text articles in Portuguese, English and Spanish from SciELO
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

"""

import os
from typing import List, Optional

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@inproceedings{soares2018large,
  title={A Large Parallel Corpus of Full-Text Scientific Articles},
  author={Soares, Felipe and Moreira, Viviane and Becker, Karin},
  booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC-2018)},
  year={2018}
}
"""

_DATASETNAME = "scielo"

_DESCRIPTION = """\
A parallel corpus of full-text scientific articles collected from Scielo database in the following languages: \
English, Portuguese and Spanish. The corpus is sentence aligned for all language pairs, \
as well as trilingual aligned for a small subset of sentences. Alignment was carried out using the Hunalign algorithm.
"""

_HOMEPAGE = "https://sites.google.com/view/felipe-soares/datasets#h.p_92uSCyAjWSRB"

_LICENSE = "CC BY 4.0"

_URLS = {
    "en-es": "https://ndownloader.figstatic.com/files/14019287",
    "en-pt": "https://ndownloader.figstatic.com/files/14019308",
    "en-pt-es": "https://ndownloader.figstatic.com/files/14019293",
}

_SUPPORTED_TASKS = [Tasks.TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class ScieloDataset(datasets.GeneratorBasedBuilder):
    """Parallel corpus of full-text articles in Portuguese, English and Spanish from SciELO."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)


    BUILDER_CONFIGS = [
        BigBioConfig(
            name="scielo_source_en-es",
            version=SOURCE_VERSION,
            description="English-Spanish",
            schema="source",
            subset_id="scielo",
        ),
        BigBioConfig(
            name="scielo_source_en-pt",
            version=SOURCE_VERSION,
            description="English-Portuguese",
            schema="source",
            subset_id="scielo",
        ),
        BigBioConfig(
            name="scielo_source_en-pt-es",
            version=SOURCE_VERSION,
            description="English-Portuguese-Spanish",
            schema="source",
            subset_id="scielo",
        ),
        BigBioConfig(
            name="scielo_bigbio_t2t_en-es",
            version=BIGBIO_VERSION,
            description="scielo BigBio schema",
            schema="bigbio_t2t",
            subset_id="scielo",
        ),
        BigBioConfig(
            name="scielo_bigbio_t2t_en-pt",
            version=BIGBIO_VERSION,
            description="scielo BigBio schema",
            schema="bigbio_t2t",
            subset_id="scielo",
        ),
    ]

    DEFAULT_CONFIG_NAME = "scielo_source_en-es"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            lang_list: List[str] = self.config.name.split("_")[-1].split("-")
            features = datasets.Features(
                    {"translation": datasets.features.Translation(languages=lang_list)}
                )

        elif self.config.schema == "bigbio_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        languages: str = self.config.name.split("_")[-1]
        archive = dl_manager.download(_URLS[languages])
        lang_list: List[str] = languages.split("-")
        fname = languages.replace("-", "_")

        if languages == "en-pt-es":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "source_file": f"{fname}.en",
                        "target_file": f"{fname}.pt",
                        "target_file_2": f"{fname}.es",
                        "files": dl_manager.iter_archive(archive),
                        "languages": languages,
                        "split": "train",
                    },
                ),
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "source_file": f"{fname}.{lang_list[0]}",
                        "target_file": f"{fname}.{lang_list[1]}",
                        "files": dl_manager.iter_archive(archive),
                        "languages": languages,
                        "split": "train",
                    },
                ),
            ]


    def _generate_examples(self, languages: str, split: str, source_file:str, target_file:str, files, target_file_2: Optional[str]=None) -> (int, dict):
        # breakpoint()
        if self.config.schema == "source":

            for path, f in files:
                if path == source_file:
                    source_sentences = f.read().decode("utf-8").split("\n")
                elif path == target_file:
                    target_sentences = f.read().decode("utf-8").split("\n")
                elif languages == "en-pt-es" and path == target_file_2:
                    target_sentences_2 = f.read().decode("utf-8").split("\n")

            if languages == "en-pt-es":
                source, target, target_2 = tuple(languages.split("-"))
                for idx, (l1, l2, l3) in enumerate(zip(source_sentences, target_sentences, target_sentences_2)):
                    result = {"translation": {source: l1, target: l2, target_2: l3}}
                    yield idx, result
            else:
                source, target = tuple(languages.split("-"))
                for idx, (l1, l2) in enumerate(zip(source_sentences, target_sentences)):
                    result = {"translation": {source: l1, target: l2}}
                    yield idx, result

        elif self.config.schema == "bigbio_t2t":
            for path, f in files:
                if path == source_file:
                    source_sentences = f.read().decode("utf-8").split("\n")
                elif path == target_file:
                    target_sentences = f.read().decode("utf-8").split("\n")

            uid = 0
            source, target = tuple(languages.split("-"))
            for idx, (l1, l2) in enumerate(zip(source_sentences, target_sentences)):
                uid += 1
                yield idx, {
                    "id": str(uid),
                    "document_id": str(idx),
                    "text_1": l1,
                    "text_2": l2,
                    "text_1_name": source,
                    "text_2_name": target,
                }



# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python scielo.py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
