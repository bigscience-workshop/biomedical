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
The DrugProt corpus consists of a) expert-labelled chemical and gene mentions, and (b) all binary relationships
between them corresponding to a specific set of biologically relevant relation types. The corpus was introduced
in context of the BioCreative VII Track 1 (Text mining drug and chemical-protein interactions).

For further information see:
https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/
"""
import collections
from pathlib import Path
from typing import Dict, Iterator, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{miranda2021overview,
  title={Overview of DrugProt BioCreative VII track: quality evaluation and large scale text mining of \
        drug-gene/protein relations},
  author={Miranda, Antonio and Mehryary, Farrokh and Luoma, Jouni and Pyysalo, Sampo and Valencia, Alfonso \
        and Krallinger, Martin},
  booktitle={Proceedings of the seventh BioCreative challenge evaluation workshop},
  year={2021}
}
"""

_DATASETNAME = "drugprot"
_DISPLAYNAME = "DrugProt"


_DESCRIPTION = """\
The DrugProt corpus consists of a) expert-labelled chemical and gene mentions, and (b) all binary relationships \
between them corresponding to a specific set of biologically relevant relation types.
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/"

_LICENSE = Licenses.CC_BY_SA_3p0

_URLS = {_DATASETNAME: "https://zenodo.org/record/5042151/files/drugprot-gs-training-development.zip?download=1"}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.2"
_BIGBIO_VERSION = "1.0.0"


class DrugProtDataset(datasets.GeneratorBasedBuilder):
    """
        The DrugProt corpus consists of a) expert-labelled chemical and gene mentions, and \
        (b) all binary relationships between them.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="drugprot_source",
            version=SOURCE_VERSION,
            description="DrugProt source schema",
            schema="source",
            subset_id="drugprot",
        ),
        BigBioConfig(
            name="drugprot_bigbio_kb",
            version=BIGBIO_VERSION,
            description="DrugProt BigBio schema",
            schema="bigbio_kb",
            subset_id="drugprot",
        ),
    ]

    DEFAULT_CONFIG_NAME = "drugprot_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offset": datasets.Sequence(datasets.Value("int32")),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
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

    def _split_generators(self, dl_manager):
        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls))
        data_dir = data_dir / "drugprot-gs-training-development"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "training"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir, "split": "development"},
            ),
        ]

    def _generate_examples(self, data_dir: Path, split: str) -> Iterator[Tuple[str, Dict]]:
        if self.config.name == "drugprot_source":
            documents = self._read_source_examples(data_dir, split)
            for document_id, document in documents.items():
                yield document_id, document

        elif self.config.name == "drugprot_bigbio_kb":
            documents = self._read_source_examples(data_dir, split)
            for document_id, document in documents.items():
                yield document_id, self._transform_source_to_kb(document)

    def _read_source_examples(self, input_dir: Path, split: str) -> Dict:
        """ """
        split_dir = input_dir / split
        abstracts_file = split_dir / f"drugprot_{split}_abstracs.tsv"
        entities_file = split_dir / f"drugprot_{split}_entities.tsv"
        relations_file = split_dir / f"drugprot_{split}_relations.tsv"

        document_to_entities = collections.defaultdict(list)
        for line in entities_file.read_text().splitlines():
            columns = line.split("\t")
            document_id = columns[0]

            document_to_entities[document_id].append(
                {
                    "id": document_id + "_" + columns[1],
                    "type": columns[2],
                    "offset": [columns[3], columns[4]],
                    "text": columns[5],
                }
            )

        document_to_relations = collections.defaultdict(list)
        for line in relations_file.read_text().splitlines():
            columns = line.split("\t")
            document_id = columns[0]

            document_relations = document_to_relations[document_id]

            document_relations.append(
                {
                    "id": document_id + "_" + str(len(document_relations)),
                    "type": columns[1],
                    "arg1_id": document_id + "_" + columns[2][5:],
                    "arg2_id": document_id + "_" + columns[3][5:],
                }
            )

        document_to_source = {}
        for line in abstracts_file.read_text().splitlines():
            document_id, title, abstract = line.split("\t")

            document_to_source[document_id] = {
                "document_id": document_id,
                "title": title,
                "abstract": abstract,
                "text": " ".join([title, abstract]),
                "entities": document_to_entities[document_id],
                "relations": document_to_relations[document_id],
            }

        return document_to_source

    def _transform_source_to_kb(self, source_document: Dict) -> Dict:
        document_id = source_document["document_id"]

        offset = 0
        passages = []
        for text_field in ["title", "abstract"]:
            text = source_document[text_field]
            passages.append(
                {
                    "id": document_id + "_" + text_field,
                    "type": text_field,
                    "text": [text],
                    "offsets": [[offset, offset + len(text)]],
                }
            )
            offset += len(text) + 1

        entities = [
            {
                "id": entity["id"],
                "type": entity["type"],
                "text": [entity["text"]],
                "offsets": [entity["offset"]],
                "normalized": [],
            }
            for entity in source_document["entities"]
        ]

        relations = [
            {
                "id": relation["id"],
                "type": relation["type"],
                "arg1_id": relation["arg1_id"],
                "arg2_id": relation["arg2_id"],
                "normalized": [],
            }
            for relation in source_document["relations"]
        ]

        return {
            "id": document_id,
            "document_id": document_id,
            "passages": passages,
            "entities": entities,
            "relations": relations,
            "events": [],
            "coreferences": [],
        }
