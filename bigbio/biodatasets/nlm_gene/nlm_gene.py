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
import collections
import itertools
from pathlib import Path

from typing import List, Tuple, Dict
from bioc import biocxml

import datasets
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks
from bigbio.utils.parsing import get_texts_and_offsets_from_bioc_ann

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@Article{islamaj2021nlm,
    title       =  {NLM-Gene, a richly annotated gold standard dataset for gene entities that addresses ambiguity and
                    multi-species gene recognition},
    author      =   {Islamaj,
                    Rezarta and
                    Wei, Chih-Hsuan and
                    Cissel, David and
                    Miliaras, Nicholas and
                    Printseva, Olga and
                    Rodionov, Oleg and
                    Sekiya, Keiko and
                    Ward, Janice and
                    Lu, Zhiyong},
    journal     =   {Journal of Biomedical Informatics},
    volume      =   {118},
    pages       =   {103779},
    year        =   {2021},
    publisher   =  {Elsevier}
}
"""

_DATASETNAME = "nlm_gene"

_DESCRIPTION = """\
NLM-Gene consists of 550 PubMed articles, from 156 journals, and contains more than 15 thousand unique gene names,
corresponding to more than five thousand gene identifiers (NCBI Gene taxonomy). This corpus contains gene annotation
data from 28 organisms. The annotated articles contain on average 29 gene names, and 10 gene identifiers per article.
These characteristics demonstrate that this article set is an important benchmark dataset to test the accuracy of gene
recognition algorithms both on multi-species and ambiguous data. The NLM-Gene corpus will be invaluable for advancing
text-mining techniques for gene identification tasks in biomedical text.
"""

_HOMEPAGE = "https://zenodo.org/record/5089049"

_LICENSE = "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication license"

_URLS = {
    "source": "https://zenodo.org/record/5089049/files/NLM-Gene-Corpus.zip",
    "bigbio_kb": "https://zenodo.org/record/5089049/files/NLM-Gene-Corpus.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class NLMGeneDataset(datasets.GeneratorBasedBuilder):
    """NLM-Gene Dataset for gene entities"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="nlm_gene_source",
            version=SOURCE_VERSION,
            description="NlM Gene source schema",
            schema="source",
            subset_id="nlm_gene",
        ),
        BigBioConfig(
            name="nlm_gene_bigbio_kb",
            version=BIGBIO_VERSION,
            description="NlM Gene BigBio schema",
            schema="bigbio_kb",
            subset_id="nlm_gene",
        ),
    ]

    DEFAULT_CONFIG_NAME = "nlm_gene_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            if self.config.schema == "source":
                # this is a variation on the BioC format
                features = datasets.Features(
                    {
                        "passages": [
                            {
                                "document_id": datasets.Value("string"),
                                "type": datasets.Value("string"),
                                "text": datasets.Value("string"),
                                "entities": [
                                    {
                                        "id": datasets.Value("string"),
                                        "offsets": [[datasets.Value("int32")]],
                                        "text": [datasets.Value("string")],
                                        "type": datasets.Value("string"),
                                        "normalized": [
                                            {
                                                "db_name": datasets.Value("string"),
                                                "db_id": datasets.Value("string"),
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
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
        """Returns SplitGenerators."""
        urls = _URLS[self.config.schema]
        data_dir = Path(dl_manager.download_and_extract(urls))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir/"Corpus",
                    "file_name": "Pmidlist.Train.txt",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir/"Corpus",
                    "file_name": "Pmidlist.Test.txt",
                    "split": "test",
                },
            ),
        ]

    @staticmethod
    def _get_bioc_entity(span, db_id_key="NCBI Gene identifier", splitters=",;|-") -> dict:
        """Parse BioC entity annotation."""
        offsets, texts = get_texts_and_offsets_from_bioc_ann(span)
        db_ids = span.infons.get(db_id_key, "-1")
        # Find connector between db_ids for the normalization, if not found, use default
        connector = "|"
        for splitter in list(splitters):
            if splitter in db_ids:
                connector = splitter
        normalized = [
            {"db_name": db_id_key, "db_id": db_id}
            for db_id in db_ids.split(connector)
        ]

        return {
            "id": span.id,
            "offsets": offsets,
            "text": texts,
            "type": span.infons["type"],
            "normalized": normalized,
        }

    def _generate_examples(self, filepath: Path, file_name: str, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath/file_name, encoding='utf-8') as f:
                contents = f.readlines()
            for uid, content in enumerate(contents):
                file_id = content.replace("\n", "")
                file_path = filepath/"FINAL"/f"{file_id}.BioC.XML"
                reader = biocxml.BioCXMLDocumentReader(file_path.as_posix())
                for xdoc in reader:
                    yield uid, {
                            "passages": [
                                {
                                    "document_id": xdoc.id,
                                    "type": passage.infons["type"],
                                    "text": passage.text,
                                    "entities": [self._get_bioc_entity(span) for span in passage.annotations],
                                }
                                for passage in xdoc.passages
                            ]
                          }
        elif self.config.schema == "bigbio_kb":
            with open(filepath/file_name, encoding='utf-8') as f:
                contents = f.readlines()
            uid = 0  # global unique id
            for i, content in enumerate(contents):
                file_id = content.replace("\n", "")
                file_path = filepath / "FINAL" / f"{file_id}.BioC.XML"
                reader = biocxml.BioCXMLDocumentReader(file_path.as_posix())
                for xdoc in reader:
                    data = {
                        "id": uid,
                        "document_id": xdoc.id,
                        "passages": [],
                        "entities": [],
                        "relations": [],
                        "events": [],
                        "coreferences": [],
                    }
                    uid += 1

                    char_start = 0
                    # passages must not overlap and spans must cover the entire document
                    for passage in xdoc.passages:
                        offsets = [[char_start, char_start + len(passage.text)]]
                        char_start = char_start + len(passage.text) + 1
                        data["passages"].append(
                            {
                                "id": uid,
                                "type": passage.infons["type"],
                                "text": [passage.text],
                                "offsets": offsets,
                            }
                        )
                        uid += 1
                    # entities
                    for passage in xdoc.passages:
                        for span in passage.annotations:
                            ent = self._get_bioc_entity(span, db_id_key="NCBI Gene identifier")
                            ent["id"] = uid  # override BioC default id
                            data["entities"].append(ent)
                            uid += 1

                    yield i, data
