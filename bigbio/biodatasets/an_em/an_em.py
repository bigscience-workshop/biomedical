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
AnEM corpus is a domain- and species-independent resource manually annotated for anatomical
entity mentions using a fine-grained classification system. The corpus consists of 500 documents
(over 90,000 words) selected randomly from citation abstracts and full-text papers with
the aim of making the corpus representative of the entire available biomedical scientific
literature. The corpus annotation covers mentions of both healthy and pathological anatomical
entities and contains over 3,000 annotated mentions.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

import bigbio.utils.parsing as parse
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{ohta-etal-2012-open,
  author    = {Ohta, Tomoko and Pyysalo, Sampo and Tsujii, Jun{'}ichi and Ananiadou, Sophia},
  title     = {Open-domain Anatomical Entity Mention Detection},
  journal   = {},
  volume    = {W12-43},
  year      = {2012},
  url       = {https://aclanthology.org/W12-4304},
  doi       = {},
  biburl    = {},
  bibsource = {},
  publisher = {Association for Computational Linguistics}
}
"""

_DATASETNAME = "an_em"
_DISPLAYNAME = "AnEM"

_DESCRIPTION = """\
AnEM corpus is a domain- and species-independent resource manually annotated for anatomical
entity mentions using a fine-grained classification system. The corpus consists of 500 documents \
(over 90,000 words) selected randomly from citation abstracts and full-text papers with \
the aim of making the corpus representative of the entire available biomedical scientific \
literature. The corpus annotation covers mentions of both healthy and pathological anatomical \
entities and contains over 3,000 annotated mentions.
"""


_HOMEPAGE = "http://www.nactem.ac.uk/anatomy/"

_LICENSE = Licenses.CC_BY_SA_3p0

_URLS = {
    _DATASETNAME: "http://www.nactem.ac.uk/anatomy/data/AnEM-1.0.4.tar.gz",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.COREFERENCE_RESOLUTION,
    Tasks.RELATION_EXTRACTION,
]

_SOURCE_VERSION = "1.0.4"
_BIGBIO_VERSION = "1.0.0"


class AnEMDataset(datasets.GeneratorBasedBuilder):
    """Anatomical Entity Mention (AnEM) corpus"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="an_em_source",
            version=SOURCE_VERSION,
            description="AnEM source schema",
            schema="source",
            subset_id="an_em",
        ),
        BigBioConfig(
            name="an_em_bigbio_kb",
            version=BIGBIO_VERSION,
            description="AnEM BigBio schema",
            schema="bigbio_kb",
            subset_id="an_em",
        ),
    ]

    DEFAULT_CONFIG_NAME = "an_em_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "document_type": datasets.Value("string"),
                    "text_type": datasets.Value("string"),
                    "entities": [
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                        }
                    ],
                    "equivalences": [
                        {
                            "entity_id": datasets.Value("string"),
                            "ref_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                    "relations": [
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
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls))
        all_data = data_dir / "AnEM-1.0.4" / "standoff"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": all_data,
                    "split_path": data_dir / "AnEM-1.0.4" / "development" / "train-files.list",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": all_data,
                    "split_path": data_dir / "AnEM-1.0.4" / "test" / "test-files.list",
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": all_data,
                    "split_path": data_dir / "AnEM-1.0.4" / "development" / "test-files.list",
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split_path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(split_path, "r") as sp:
            split_list = [line.rstrip() for line in sp]

        if self.config.schema == "source":
            for file in filepath.iterdir():

                # Use brat text files and consider files in the provided split list
                if (file.suffix != ".txt") or (file.stem not in split_list):
                    continue
                brat_parsed = parse.parse_brat_file(file)
                source_example = self._brat_to_source(file, brat_parsed)

                yield source_example["document_id"], source_example

        elif self.config.schema == "bigbio_kb":
            for file in filepath.iterdir():

                # Use brat text files and consider files in the provided split list
                if (file.suffix != ".txt") or (file.stem not in split_list):
                    continue
                brat_parsed = parse.parse_brat_file(file)
                bigbio_kb_example = parse.brat_parse_to_bigbio_kb(brat_parsed)

                bigbio_kb_example["id"] = bigbio_kb_example["document_id"]

                doc_type, text_type = self.get_document_type_and_text_type(file)
                bigbio_kb_example["passages"][0]["type"] = text_type

                yield bigbio_kb_example["id"], bigbio_kb_example

    def _brat_to_source(self, filepath, brat_example):
        """
        Converts parsed brat example to source schema example
        """
        document_type, text_type = self.get_document_type_and_text_type(filepath)

        source_example = {
            "document_id": brat_example["document_id"],
            "text": brat_example["text"],
            "document_type": document_type,
            "text_type": text_type,
            "entities": [
                {
                    "offsets": brat_entity["offsets"],
                    "text": brat_entity["text"],
                    "type": brat_entity["type"],
                    "entity_id": f"{brat_example['document_id']}_{brat_entity['id']}",
                }
                for brat_entity in brat_example["text_bound_annotations"]
            ],
            "equivalences": [
                {
                    "entity_id": brat_entity["id"],
                    "ref_ids": [f"{brat_example['document_id']}_{ids}" for ids in brat_entity["ref_ids"]],
                }
                for brat_entity in brat_example["equivalences"]
            ],
            "relations": [
                {
                    "id": f"{brat_example['document_id']}_{brat_entity['id']}",
                    "head": {
                        "ref_id": f"{brat_example['document_id']}_{brat_entity['head']['ref_id']}",
                        "role": brat_entity["head"]["role"],
                    },
                    "tail": {
                        "ref_id": f"{brat_example['document_id']}_{brat_entity['tail']['ref_id']}",
                        "role": brat_entity["tail"]["role"],
                    },
                    "type": brat_entity["type"],
                }
                for brat_entity in brat_example["relations"]
            ],
        }

        return source_example

    def get_document_type_and_text_type(self, input_file: Path) -> Tuple[str, str]:
        """
        Implementation used from
        https://github.com/bigscience-workshop/biomedical/blob/master/biodatasets/anat_em/anat_em.py

        Extracts the document type (PubMed(PM) or PubMedCentral (PMC)) and the respective
        text type (abstract for PM and sec or caption for (PMC) from the name of the given
        file, e.g.:

        PMID-9778569.txt -> ("PM", "abstract")

        PMC-1274342-sec-02.txt -> ("PMC", "sec")

        PMC-1592597-caption-02.ann -> ("PMC", "caption")

        """
        name_parts = str(input_file.stem).split("-")

        if name_parts[0] == "PMID":
            return "PM", "abstract"

        elif name_parts[0] == "PMC":
            return "PMC", name_parts[2]
        else:
            raise AssertionError(f"Unexpected file prefix {name_parts[0]}")
