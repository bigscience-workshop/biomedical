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
"""This script was originally authored by Jason Fries"""


import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import bioc
import datasets

_CITATION = """\
@Article{islamaj2021nlm,
title={NLM-Chem, a new resource for chemical entity recognition in PubMed full text literature},
author={Islamaj, Rezarta and Leaman, Robert and Kim, Sun and Kwon, Dongseop and Wei, Chih-Hsuan and Comeau, Donald C and Peng, Yifan and Cissel, David and Coss, Cathleen and Fisher, Carol and others},
journal={Scientific Data},
volume={8},
number={1},
pages={1--12},
year={2021},
publisher={Nature Publishing Group}
}
"""

_DESCRIPTION = """\
NLM-Chem corpus consists of 150 full-text articles from the PubMed Central Open Access dataset,
comprising 67 different chemical journals, aiming to cover a general distribution of usage of chemical
names in the biomedical literature.
Articles were selected so that human annotation was most valuable (meaning that they were rich in bio-entities,
and current state-of-the-art named entity recognition systems disagreed on bio-entity recognition.
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-2"

_LICENSE = " CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"

# files found here `https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/` have issues at extraction
_URLs = {"biocreative": "https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem"}


@dataclass
class Entity:
    """Entity container"""

    offsets: List[Tuple[int, int]]
    text: str
    type: str
    entity_id: str

    def to_dict(self) -> dict:
        """Dicitonary view"""
        return {
            "offsets": self.offsets,
            "text": self.text,
            "type": self.type,
            "entity_id": self.entity_id,
        }


@dataclass
class Example:
    """Example container"""

    article_id: str
    text: str
    entities: List[Entity]

    def to_dict(self) -> dict:
        """Dicitonary view"""
        return {
            "article_id": self.article_id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
        }


class NLMChemDataset(datasets.GeneratorBasedBuilder):
    """NLMChem"""

    # v2 here: https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/
    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="biocreative",
            version=VERSION,
            description="Original annotation files.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "biocreative"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):

        if self.config.name == "biocreative":
            features = datasets.Features(
                {
                    "passages": datasets.Sequence(
                        {
                            "pmid": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "entities": datasets.Sequence(
                                {
                                    "offsets": datasets.Sequence(
                                        [datasets.Value("int32")]
                                    ),
                                    "text": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                    "entity_id": datasets.Value("string"),
                                }
                            ),
                        }
                    )
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "dev",
                },
            ),
        ]

    def _get_article_ids_from_split(self, data_dir: str, split: str) -> Iterable[str]:
        split_file = os.path.join(data_dir, "FINAL_v1", f"pmcids_{split}.txt")
        article_ids = open(split_file).readlines()
        article_ids = [a.strip() for a in article_ids]

        return article_ids

    def _build_example(self, article_id: str, passage: bioc.BioCPassage) -> Example:
        return Example(
            article_id=article_id,
            text=passage.text,
            entities=[self._get_bioc_entity(s) for s in passage.annotations],
        )

    def _get_bioc_entity(self, span: bioc.BioCAnnotation) -> Entity:
        offsets: List[Tuple[int, int]] = [
            (loc.offset, loc.offset + loc.length) for loc in span.locations
        ]
        return Entity(
            offsets=offsets,
            text=span.text,
            type=span.infons.get("type"),
            entity_id=span.infons.get("identifier"),
        )

    def _load_collection(self, data_dir: str, article_id: str) -> bioc.BioCCollection:

        path = os.path.join(data_dir, "FINAL_v1", "ALL", f"{article_id}_v1.xml")

        return bioc.load(path)

    def _generate_examples(
        self,
        data_dir,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""
        if self.config.name == "biocreative":
            idx = 0
            article_ids = self._get_article_ids_from_split(
                data_dir=data_dir, split=split
            )
            for article_id in article_ids:
                colleciton = self._load_collection(
                    data_dir=data_dir, article_id=article_id
                )
                for d in colleciton.documents:
                    for p in d.passages:
                        example = self._build_example(passage=p, article_id=article_id)
                        yield (idx, example.to_dict())
                        idx += 1
