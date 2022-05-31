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
LINNAEUS provides a novel corpus of full-text documents manually annotated for species mentions.

To understand the true performance of the LINNAEUS system, we generated a gold standard dataset specifically
annotated to evaluate species name identification software. The reliability of this gold standard is high,
however some species names are likely to be omitted from this evaluation set, as shown by IAA analysis.
Performance of species tagging by LINNAEUS on full-text articles is very good, with 94.3% recall and
97.1% precision on mention level, and 98.1% recall and 90.4% precision on document level.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LOCAL = False
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

_LICENSE = Licenses.CC_BY_4p0

_URLS = {
    _DATASETNAME: "https://sourceforge.net/projects/linnaeus/files/Corpora/manual-corpus-species-1.0.tar.gz/download",
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
            description="Linnaeus source schema",
            schema="source",
            subset_id="linnaeus",
        ),
        BigBioConfig(
            name="linnaeus_filtered_source",
            version=SOURCE_VERSION,
            description="Linnaeus source schema (filtered tags)",
            schema="source",
            subset_id="linnaeus_filtered",
        ),
        BigBioConfig(
            name="linnaeus_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Linnaeus BigBio schema",
            schema="bigbio_kb",
            subset_id="linnaeus",
        ),
        BigBioConfig(
            name="linnaeus_filtered_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Linnaeus BigBio schema (filtered tags)",
            schema="bigbio_kb",
            subset_id="linnaeus_filtered",
        ),
    ]

    DEFAULT_CONFIG_NAME = "linneaus_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "document_type": datasets.Value("string"),
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
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
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
        """Yields examples as (key, example) tuples."""
        data_path = Path(os.path.join(data_files, "txt"))
        if self.config.subset_id.endswith("filtered"):
            tags_path = Path(os.path.join(data_files, "filtered_tags.tsv"))
        else:
            tags_path = Path(os.path.join(data_files, "tags.tsv"))
        data_files = list(data_path.glob("*txt"))
        tags = self._load_tags(tags_path)

        if self.config.schema == "source":
            for guid, data_file in enumerate(data_files):
                document_key = data_file.stem
                if document_key not in tags:
                    continue
                example = self._create_source_example(data_file, tags.get(document_key))
                example["document_id"] = str(document_key)
                yield guid, example

        elif self.config.schema == "bigbio_kb":
            for guid, data_file in enumerate(data_files):
                document_key = data_file.stem
                if document_key not in tags:
                    continue
                example = self._create_kb_example(data_file, tags.get(document_key))
                example["document_id"] = str(document_key)
                example["id"] = guid
                yield guid, example

    @staticmethod
    def _load_tags(path: Path) -> Dict:
        """Loads all tags into a dictionary with document ID as keys and all annotations to that file as values."""
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

    def _create_source_example(self, txt_file, tags) -> Dict:
        """Creates example in source schema."""
        example = {}
        example["entities"] = []
        with open(txt_file, "r") as file:
            text = file.read()
        example["text"] = text
        example["document_type"] = "Article"
        for tag_id, tag in enumerate(tags):
            species_id, start, end, entity_text, _ = tag
            entity_type, db_name, db_id = species_id.split(":")
            entity = {
                "id": str(tag_id),
                "type": entity_type,
                "text": [entity_text],
                "offsets": [(int(start), int(end))],
                "normalized": [
                    {
                        "db_name": db_name,
                        "db_id": db_id,
                    }
                ],
            }
            example["entities"].append(entity)
        return example

    def _create_kb_example(self, txt_file, tags) -> Dict:
        """Creates example in BigBio KB schema."""
        example = {}
        with open(txt_file, "r") as file:
            text = file.read()
        # Passages
        example["passages"] = [
            {
                "id": f"{txt_file.stem}__text",
                "text": [text],
                "type": "Article",
                "offsets": [(0, len(text))],
            }
        ]
        # Entities
        example["entities"] = []
        for tag_id, tag in enumerate(tags):
            species_id, start, end, entity_text, _ = tag
            entity_type, db_name, db_id = species_id.split(":")
            entity = {
                "id": f"{txt_file.stem}__T{str(tag_id)}",
                "type": entity_type,
                "text": [entity_text],
                "offsets": [(int(start), int(end))],
                "normalized": [
                    {
                        "db_name": db_name,
                        "db_id": db_id,
                    }
                ],
            }
            example["entities"].append(entity)
        example["events"] = []
        example["relations"] = []
        example["coreferences"] = []
        return example
