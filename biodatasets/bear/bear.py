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

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@InProceedings{wuehrl_klinger_2022,
  author    = {Wuehrl, Amelie  and  Klinger, Roman},
  title     = {Recovering Patient Journeys: A Corpus of Biomedical Entities and Relations on  Twitter (BEAR)},
  booktitle      = {Proceedings of The 13th Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association}
}
"""

_DATASETNAME = "bear"

_DESCRIPTION = """\
A dataset of 2100 Twitter posts annotated with 14 different types of biomedical entities (e.g., disease, treatment,
risk factor, etc.) and 20 relation types (including caused, treated, worsens, etc.).
"""

_HOMEPAGE = "https://www.ims.uni-stuttgart.de/en/research/resources/corpora/bioclaim/"
_LICENSE = "CC BY-SA"

_URLS = {
    _DATASETNAME: "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/bioclaim/bear-corpus-WuehrlKlinger-\
LREC2022.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BearDataset(datasets.GeneratorBasedBuilder):
    """
    BEAR: A Corpus of Biomedical Entities and Relations

    A dataset of 2100 Twitter posts annotated with 14 different types of
    biomedical entities (e.g., disease, treatment, risk factor, etc.)
    and 20 relation types (including caused, treated, worsens, etc.).
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bear_source",
            version=SOURCE_VERSION,
            description="bear source schema",
            schema="source",
            subset_id="bear",
        ),
        BigBioConfig(
            name="bear_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bear BigBio schema",
            schema="bigbio_kb",
            subset_id="bear",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bear_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "document_text": datasets.Value("string"),
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": datasets.Sequence(datasets.Value("int32")),
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
                gen_kwargs={
                    "filepath": Path(data_dir) / "corpus" / "bear.jsonl",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        uid = 0
        input_file = filepath
        with open(input_file, "r") as file:
            for line in file:
                document: Dict = json.loads(line)

                document_id: str = document.pop("doc_id")
                document_text: str = document.pop("doc_text")
                entities: Dict[str, Dict[str, Union[str, int]]] = document.pop("entities", {})
                relations: List[Dict[str, Union[str, int]]] = document.pop("relations", [])

                if not entities and not relations:
                    continue

                if self.config.schema == "source":
                    source_example = self._to_source_example(
                        document_id=document_id,
                        document_text=document_text,
                        entities=entities,
                        relations=relations,
                    )
                    yield uid, source_example
                elif self.config.schema == "bigbio_kb":
                    bigbio_example = self._to_bigbio_example(
                        document_id=document_id,
                        document_text=document_text,
                        entities=entities,
                        relations=relations,
                    )
                    yield uid, bigbio_example

                uid += 1

    def _to_source_example(
        self,
        document_id: str,
        document_text: str,
        entities: Dict[str, Dict[str, Union[str, int]]],
        relations: List[Dict[str, Union[str, int]]],
    ) -> Dict:
        source_example = {
            "document_id": document_id,
            "document_text": document_text,
        }

        # Capture Entities
        _entities = []
        for id, entity_values in entities.items():
            if not entity_values:
                continue
            start = entity_values.pop("begin")
            end = entity_values.pop("end")
            type = entity_values.pop("tag")
            text = document_text[start:end]

            entity = {
                "id": f"{document_id}_{id}",
                "type": type,
                "text": text,
                "offsets": [start, end],
            }
            _entities.append(entity)
        source_example["entities"] = _entities

        # Capture Relations
        _relations = []
        for id, relation_values in enumerate(relations):
            if not relation_values:
                continue
            end_entity = relation_values.pop("end_entity")
            rel_tag = relation_values.pop("rel_tag")
            start_entity = relation_values.pop("start_entity")

            relation = {
                "id": f"{document_id}_relation_{id}",
                "type": rel_tag,
                "arg1_id": f"{document_id}_{start_entity}",
                "arg2_id": f"{document_id}_{end_entity}",
            }
            _relations.append(relation)
        source_example["relations"] = _relations

        return source_example

    def _to_bigbio_example(
        self,
        document_id: str,
        document_text: str,
        entities: Dict[str, Dict[str, Union[str, int]]],
        relations: List[Dict[str, Union[str, int]]],
    ) -> Dict:
        bigbio_example = {
            "id": f"{document_id}_id",
            "document_id": document_id,
            "passages": [
                {
                    "id": f"{document_id}_passage",
                    "type": "social_media_text",
                    "text": [document_text],
                    "offsets": [[0, len(document_text)]],
                }
            ],
            "events": [],
            "coreferences": [],
        }

        # Capture Entities
        _entities = []
        for id, entity_values in entities.items():
            if not entity_values:
                continue
            start = entity_values.pop("begin")
            end = entity_values.pop("end")
            type = entity_values.pop("tag")
            text = document_text[start:end]

            entity = {
                "id": f"{document_id}_{id}",
                "type": type,
                "text": [text],
                "offsets": [[start, end]],
                "normalized": [],
            }
            _entities.append(entity)
        bigbio_example["entities"] = _entities

        # Capture Relations
        _relations = []
        for id, relation_values in enumerate(relations):
            if not relation_values:
                continue
            end_entity = relation_values.pop("end_entity")
            rel_tag = relation_values.pop("rel_tag")
            start_entity = relation_values.pop("start_entity")

            relation = {
                "id": f"{document_id}_relation_{id}",
                "type": rel_tag,
                "arg1_id": f"{document_id}_{start_entity}",
                "arg2_id": f"{document_id}_{end_entity}",
                "normalized": [],
            }
            _relations.append(relation)
        bigbio_example["relations"] = _relations

        return bigbio_example
