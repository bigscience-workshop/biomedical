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
S800 Corpus: a novel abstract-based manually annotated corpus for Named Entity Recognition.
S800 comprises 800 PubMed abstracts in which organism mentions were identified and mapped
to the corresponding NCBI Taxonomy identifiers.

To increase the corpus taxonomic mention diversity the S800 abstracts were collected by
selecting 100 abstracts from the following 8 categories:
bacteriology, botany, entomology, medicine, mycology, protistology, virology and zoology.
S800 has been annotated with a focus at the species level;
however, higher taxa mentions (such as genera, families and orders) have also been considered.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
import pandas as pd

from .bigbiohub import BigBioConfig, Tasks, kb_features

_LOCAL = False

_CITATION = """\
@article{
    title = {The SPECIES and ORGANISMS Resources for Fast and Accurate Identification of Taxonomic Names in Text},
    author = {Pafilis, Evangelos AND Frankild, Sune P. AND Fanini,
        Lucia AND Faulwetter, Sarah AND Pavloudi, Christina AND Vasileiadou,
        Aikaterini AND Arvanitidis, Christos AND Jensen, Lars Juhl},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    year = {2013},
    month = {06},
    volume = {8},
    pages = {1-6},
    number = {6},
    url = {https://doi.org/10.1371/journal.pone.0065390},
    doi = {10.1371/journal.pone.0065390},
}
"""

_DATASETNAME = "s800"

_DISPLAYNAME = "S800"

_DESCRIPTION = """\
S800 Corpus: a novel abstract-based manually annotated corpus for Named Entity Recognition.
S800 comprises 800 PubMed abstracts in which organism mentions were identified and mapped
to the corresponding NCBI Taxonomy identifiers.

To increase the corpus taxonomic mention diversity the S800 abstracts were collected by
selecting 100 abstracts from the following 8 categories:
bacteriology, botany, entomology, medicine, mycology, protistology, virology and zoology.
S800 has been annotated with a focus at the species level;
however, higher taxa mentions (such as genera, families and orders) have also been considered.
"""

_HOMEPAGE = "https://species.jensenlab.org/"

_LICENSE = "OTHER"  # "subject to Medline restrictions"

_URLS = {
    _DATASETNAME: "https://species.jensenlab.org/files/S800-1.0.tar.gz",
}

_LANGUAGES = ["English"]

_PUBMED = True

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class S800Dataset(datasets.GeneratorBasedBuilder):
    """S800 comprises 800 PubMed abstracts in which organism mentions
    were identified and mapped to the corresponding NCBI Taxonomy identifiers."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        BigBioConfig(
            name=f"{_DATASETNAME}_bigbio_kb",
            version=BIGBIO_VERSION,
            description=f"{_DATASETNAME} BigBio schema",
            schema="bigbio_kb",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "s800_doc_id": datasets.Value("string"),
                    "pmid": datasets.Value("string"),
                    "entities": [
                        {
                            "offsets": datasets.Sequence(datasets.Value("int64")),
                            "text": datasets.Value("string"),
                            "ncbi_txid": datasets.Value("string"),
                        }
                    ],
                    "category": datasets.Value("string"),
                    "category_id": datasets.Value("int64"),
                    "journal": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": Path(data_dir),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, data_dir: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            for key, example in self._read_example_from_file(data_dir):
                yield key, example

        elif self.config.schema == "bigbio_kb":
            for key, example in self._read_example_from_file_in_kb_schema(data_dir):
                yield key, example

    def _read_example_from_file(self, data_dir: Path) -> Tuple[str, Dict]:
        abstract_dir = data_dir / "abstracts"
        df_s800 = pd.read_csv(
            data_dir / "S800.tsv",
            sep="\t",
            header=None,
            names=["nbci_taxonomy_id", "doc_id", "start", "end", "phrase"],
        )
        df_s800["s800_doc_id"] = df_s800["doc_id"].apply(lambda x: x.split(":")[0])

        df_pubmed = pd.read_csv(
            data_dir / "pubmedid.tsv",
            sep="\t",
            header=None,
            names=["s800_doc_id", "pmid", "category", "category_id", "journal"],
        )
        for _, row in df_pubmed.iterrows():
            key = row.s800_doc_id
            entities = [
                dict(
                    offsets=[entity_row.start, entity_row.end],
                    text=entity_row.phrase,
                    ncbi_txid=entity_row.nbci_taxonomy_id,
                )
                for _, entity_row in df_s800[df_s800.s800_doc_id == key].iterrows()
            ]
            doc_abstract_path = abstract_dir / f"{key}.txt"
            with open(doc_abstract_path, encoding="utf-8") as fp:
                text = fp.read()
            example = {
                "doc_id": key,
                "s800_doc_id": key,
                "pmid": row.pmid,
                "entities": entities,
                "category": row.category,
                "category_id": row.category_id,
                "journal": row.journal,
                "text": text,
            }
            yield key, example

    def _parse_example_to_kb_schema(self, example) -> Dict[str, Any]:
        text = example["text"]
        doc_id = example["doc_id"]
        passages = [
            {
                "id": f"{doc_id}-P0",
                "type": "abstract",
                "text": [text],
                "offsets": [[0, len(text)]],
            }
        ]
        entities = []
        for i, entity in enumerate(example["entities"]):
            cs, ce = entity["offsets"]
            ce = ce + 1  # Add 1 to make the offset exclusive
            entity = {
                "id": f"{doc_id}-E{i}",
                "text": [entity["text"]],
                "offsets": [[cs, ce]],
                "type": "species",
                "normalized": [{"db_id": entity["ncbi_txid"], "db_name": "NBCI Taxonomy"}],
            }
            entities.append(entity)
        data = {
            "id": doc_id,
            "document_id": doc_id,
            "passages": passages,
            "entities": entities,
            "relations": [],
            "events": [],
            "coreferences": [],
        }
        return data

    def _read_example_from_file_in_kb_schema(self, data_dir: Path) -> Tuple[str, Dict]:
        for key, example in self._read_example_from_file(data_dir):
            example = self._parse_example_to_kb_schema(example)
            yield key, example
