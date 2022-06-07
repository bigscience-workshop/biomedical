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

"""Descriptions of genetic variations and their effect are widely spread across the biomedical literature. However,
finding all mentions of a specific variation, or all mentions of variations in a specific gene, is difficult to
achieve due to the many ways such variations are described. Here, we describe SETH, a tool for the recognition of
variations from text and their subsequent normalization to dbSNP or UniProt. SETH achieves high precision and recall
on several evaluation corpora of PubMed abstracts. It is freely available and encompasses stand-alone scripts for
isolated application and evaluation as well as a thorough documentation for integration into other applications.
The script loads dataset in bigbio schema (using knowledgebase schema: schemas/kb) AND/OR source (default) schema """

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = []
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@Article{SETH2016,
    Title       = {SETH detects and normalizes genetic variants in text.},
    Author      = {Thomas, Philippe and Rockt{"{a}}schel, Tim and Hakenberg, J{"{o}}rg and Lichtblau, Yvonne and Leser, Ulf},
    Journal     = {Bioinformatics},
    Year        = {2016},
    Month       = {Jun},
    Doi         = {10.1093/bioinformatics/btw234},
    Language    = {eng},
    Medline-pst = {aheadofprint},
    Pmid        = {27256315},
    Url         = {http://dx.doi.org/10.1093/bioinformatics/btw234
}
"""

_DATASETNAME = "seth_corpus"

_DESCRIPTION = (
    """SNP named entity recognition corpus consisting of 630 PubMed citations."""
)

_HOMEPAGE = "https://github.com/rockt/SETH"

_LICENSE = Licenses.APACHE_2p0
_URLS = {
    "source": "https://github.com/rockt/SETH/archive/refs/heads/master.zip",
    "bigbio_kb": "https://github.com/rockt/SETH/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class SethCorpusDataset(datasets.GeneratorBasedBuilder):
    """SNP named entity recognition corpus consisting of 630 PubMed citations."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="seth_corpus_source",
            version=SOURCE_VERSION,
            description="SETH corpus source schema",
            schema="source",
            subset_id="seth_corpus",
        ),
        BigBioConfig(
            name="seth_corpus_bigbio_kb",
            version=BIGBIO_VERSION,
            description="SETH corpus BigBio schema",
            schema="bigbio_kb",
            subset_id="seth_corpus",
        ),
    ]

    DEFAULT_CONFIG_NAME = "seth_corpus_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
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
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[self.config.schema]
        data_dir = Path(dl_manager.download_and_extract(urls))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir / "SETH-master" / "resources" / "SETH-corpus",
                    "corpus_file": "corpus.txt",
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(
        self, filepath: Path, corpus_file: str, split: str
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath / corpus_file, encoding="utf-8") as f:
                contents = f.readlines()
            for guid, content in enumerate(contents):
                file_name, text = content.split("\t")
                example = parsing.parse_brat_file(
                    filepath / "annotations" / f"{file_name}.ann"
                )
                example["id"] = str(guid)
                example["text"] = text
                yield guid, example

        elif self.config.schema == "bigbio_kb":
            with open(filepath / corpus_file, encoding="utf-8") as f:
                contents = f.readlines()
            for guid, content in enumerate(contents):
                file_name, text = content.split("\t")
                example = parsing.parse_brat_file(
                    filepath / "annotations" / f"{file_name}.ann"
                )

                # this example contains event lines
                # but events have not arguments
                # this is most likely an error on the annotation side
                if example["document_id"] == "11058905":
                    example["events"] = []

                example["text"] = text
                example = parsing.brat_parse_to_bigbio_kb(example)
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
