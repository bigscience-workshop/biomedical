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
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

[bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""

import os
import csv
from pathlib import Path
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@Article{gerner2010linnaeus,
title={LINNAEUS: a species name identification system for biomedical literature},
author={Gerner, Martin and Nenadic, Goran and Bergman, Casey M},
journal={BMC bioinformatics},
volume={11},
number={1},
pages={1--17},
year={2010},
publisher={BioMed Central}
}
"""

_DATASETNAME = "linnaeus"

_DESCRIPTION = """\
Linnaeus is a novel corpus of full-text documents manually annotated for species mentions.
"""

_HOMEPAGE = "http://linnaeus.sourceforge.net/"

_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

_URLS = {
    "source": "https://sourceforge.net/projects/linnaeus/files/Corpora/manual-corpus-species-1.0.tar.gz/download",
    "bigbio_text": "https://sourceforge.net/projects/linnaeus/files/Corpora/manual-corpus-species-1.0.tar.gz/download",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class LinnaeusDataset(datasets.GeneratorBasedBuilder):
    """Linneaus provides a new gold-standard corpus of full-text articles
    with manually annotated mentions of species names."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="linnaeus_source",
            version=SOURCE_VERSION,
            description="linnaeus source schema",
            schema="source",
            subset_id="linnaeus",
        ),
        BigBioConfig(
            name="linnaeus_bigbio_kb",
            version=BIGBIO_VERSION,
            description="linnaeus BigBio schema",
            schema="bigbio_kb",
            subset_id="linnaeus",
        ),
    ]

    DEFAULT_CONFIG_NAME = "linneaus_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
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
        urls = _URLS[self.config.schema]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_files": os.path.join(data_dir, "manual-corpus-species-1.0")
                },
            ),
        ]

    def _generate_examples(self, data_files: Path) -> Tuple[int, Dict]:
        data_path = Path(os.path.join(data_files, "txt"))
        tags_path = Path(os.path.join(data_files, "tags.tsv"))

        if self.config.schema == "source":
            data_files = list(data_path.glob("*txt"))
            tags = self._load_tags(tags_path)
            for guid, data_file in enumerate(data_files):
                document_key = self._extract_document_key(data_file)
                example = self._create_example(data_file, tags.get(document_key))
                example["document_id"] = str(document_key)
                example["id"] = str(guid)
                breakpoint()
                yield guid, example

        elif self.config.schema == "bigbio_[bigbio_schema_name]":
            pass
            # TODO: yield (key, example) tuples in the bigbio schema
            #for key, example in thing:
            #    yield key, example

    @staticmethod
    def _load_tags(path: Path) -> Dict:
        """
        This method loads all tags into a dictionary with document ID as keys and all annotations to that file as values.
        """
        tags = {}
        document_id_col = 1

        with open(path, encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="\t")
            next(reader)
            for line in reader:
                document_id = line[document_id_col]
                line.pop(document_id_col)
                if document_id not in tags:
                    tags[document_id] = [line]
                else:
                    tags[document_id].append(line)
        return tags

    @staticmethod
    def _extract_document_key(path: Path):
        return path.stem

    def _create_example(self, txt_file, tags) -> Dict:
        example = {}
        example["entities"] = []
        with open(txt_file, 'r') as file:
            text = file.read()
        example["text"] = text
        for tag_id, tag in enumerate(tags):
            species_id, start, end, entity_text, normalized_entity_id = tag
            entity = {
                "id": tag_id,
                "type": species_id,
                "text": entity_text,
                "offsets": [int(start), int(end)],
                "normalized": [{
                    "db_name": self._normalized_entity_lookup(normalized_entity_id),
                    "db_id": None if normalized_entity_id == "" else normalized_entity_id,
                }]
            }
            example["entities"].append(entity)
        return example

    @staticmethod
    def _normalized_entity_lookup(normalized_entity_id: str) -> str:
        normalized_entities = {
            "0": "misspelling",
            "1": "OCR error (caused by OCR software when scanning articles into text format)",
            "2": "incorrect name usage (e.g. using Drosophila when referring to D. melanogaster)",
            "3": "enumeration (e.g. 'V. vulnificus CMCP6 and YJ016', will result in two mentions)",
            "4": "modifier (e.g. human in 'human studies', mouse in 'mouse gene', rat in 'rat brain')",
            "20": "wrong case (e.g. 'drosophila melanogaster')"
        }

        if normalized_entity_id in normalized_entities:
            return normalized_entities.get(normalized_entity_id)
        else:
            return None



# This allows you to run your dataloader with `python [dataset_name].py` during development
## TODO: Remove this before making your PR
#if __name__ == "__main__":
#    datasets.load_dataset(__file__)