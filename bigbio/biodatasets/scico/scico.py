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

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import jsonlines

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_CITATION = """\
@inproceedings{cattan2021scico,
title={SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts},
author={Arie Cattan and Sophie Johnson and Daniel S. Weld and Ido Dagan and Iz Beltagy and Doug Downey and Tom Hope},
booktitle={3rd Conference on Automated Knowledge Base Construction},
year={2021},
url={https://openreview.net/forum?id=OFLbgUP04nC}
}
"""

_DATASETNAME = "scico"
_DESCRIPTION = """\
Hierarchical cross-document coreference resolution.
"""

_HOMEPAGE = "https://scico.apps.allenai.org"
_LICENSE = "Apache License 2.0"

_URLS = {
    _DATASETNAME: "https://nlp.biu.ac.il/~ariecattan/scico/data.tar",
}

_SUPPORTED_TASKS = [Tasks.COREFERENCE_RESOLUTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class ScicoDataset(datasets.GeneratorBasedBuilder):
    """
    SciCo

    SciCo is a dataset for hierarchical cross-document coreference resolution over scientific papers in the CS domain.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="scico_source",
            version=SOURCE_VERSION,
            description="scico source schema",
            schema="source",
            subset_id="scico",
        ),
        BigBioConfig(
            name="scico_bigbio_kb",
            version=BIGBIO_VERSION,
            description="scico BigBio schema",
            schema="bigbio_kb",
            subset_id="scico",
        ),
    ]

    DEFAULT_CONFIG_NAME = "scico_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "flatten_tokens": datasets.features.Sequence(datasets.features.Value("string")),
                    "flatten_mentions": datasets.features.Sequence(
                        datasets.features.Sequence(datasets.features.Value("int32"), length=3)
                    ),
                    "tokens": datasets.features.Sequence(
                        datasets.features.Sequence(datasets.features.Value("string"))
                    ),
                    "doc_ids": datasets.features.Sequence(datasets.features.Value("int32")),
                    "metadata": datasets.features.Sequence(
                        {
                            "title": datasets.features.Value("string"),
                            "paper_sha": datasets.features.Value("string"),
                            "fields_of_study": datasets.features.Value("string"),
                            "Year": datasets.features.Value("string"),
                            "BookTitle": datasets.features.Value("string"),
                            "url": datasets.features.Value("string"),
                        }
                    ),
                    "sentences": datasets.features.Sequence(
                        datasets.features.Sequence(datasets.features.Sequence(datasets.features.Value("int32")))
                    ),
                    "mentions": datasets.features.Sequence(
                        datasets.features.Sequence(datasets.features.Value("int32"), length=4)
                    ),
                    "relations": datasets.features.Sequence(
                        datasets.features.Sequence(datasets.features.Value("int32"), length=2)
                    ),
                    "id": datasets.Value("int32"),
                    "source": datasets.Value("string"),
                    "hard_10": datasets.features.Value("bool"),
                    "hard_20": datasets.features.Value("bool"),
                    "curated": datasets.features.Value("bool"),
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
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": Path(data_dir) / "train.jsonl"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": Path(data_dir) / "dev.jsonl"}
            ),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": Path(data_dir) / "test.jsonl"}),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with jsonlines.open(filepath, "r") as file:
            for uid, sample in enumerate(file):
                if self.config.schema == "source":
                    source_example = deepcopy(sample)
                    source_example["hard_10"] = source_example["hard_10"] if "hard_10" in source_example else False
                    source_example["hard_20"] = source_example["hard_20"] if "hard_20" in source_example else False
                    source_example["curated"] = source_example["curated"] if "curated" in source_example else False
                    yield uid, source_example

                elif self.config.schema == "bigbio_kb":
                    bigbio_example = self._to_bigbio_example(sample)
                    yield uid, bigbio_example

    def find_sub_list_index(self, super_list: List[str], sub_list: List[str]) -> int:
        sub_list_length = len(sub_list)
        possible_start_indices = [i for i, token in enumerate(super_list) if token == sub_list[0]]
        for index in possible_start_indices:
            if super_list[index: (index + sub_list_length)] == sub_list:
                return index

    def _to_bigbio_example(self, sample: Dict) -> Dict:

        id: int = sample.pop("id")
        source: str = sample.pop("source")

        doc_ids: List[int] = sample.pop("doc_ids")
        doc_ids = sorted(list(set(doc_ids)))
        flatten_tokens: List[str] = sample.pop("flatten_tokens")
        tokens: List[List[str]] = sample.pop("tokens")

        mentions: List[List[int]] = sample.pop("mentions")
        relations: List[List[int]] = sample.pop("relations")

        bigbio_example = {
            "id": f"{id}",
            "document_id": f"document_{id}",
            "events": [],
        }

        passages = []
        for passage_doc_id, sentence_tokens in zip(doc_ids, tokens):
            passage_text = " ".join(sentence_tokens)

            start = self.find_sub_list_index(flatten_tokens, sentence_tokens)
            passage_offset_start = len(" ".join(flatten_tokens[0:start]))
            if passage_offset_start > 0:
                passage_offset_start += 1
            passage_offset_end = passage_offset_start + len(passage_text)
            passages.append(
                {
                    "id": f"{id}_{passage_doc_id}_passage",
                    "type": source,
                    "text": [passage_text],
                    "offsets": [[passage_offset_start, passage_offset_end]],
                }
            )

        entities = []
        corefs: Dict[str, List[str]] = {}
        for i, mention in enumerate(mentions):
            mention_doc_id, start, end, cluster_id = mention

            entity_id = f"{id}_doc_{mention_doc_id}_entity_{i}"
            cluster_id = f"{id}_cluster_{cluster_id}"
            entity_text = " ".join(flatten_tokens[start: (end + 1)])
            entity_offset_start = len(" ".join(flatten_tokens[0:start]))
            if entity_offset_start > 0:
                entity_offset_start += 1
            entity_offset_end = entity_offset_start + len(entity_text)
            entities.append(
                {
                    "id": entity_id,
                    "type": cluster_id,
                    "text": [entity_text],
                    "offsets": [[entity_offset_start, entity_offset_end]],
                    "normalized": [],
                }
            )

            if cluster_id not in corefs:
                corefs[cluster_id] = []
            corefs[cluster_id].append(entity_id)

        coreferences = []
        for coref_cluster_id, entity_ids in corefs.items():
            coreferences.append(
                {
                    "id": coref_cluster_id,
                    "entity_ids": entity_ids,
                }
            )

        relations = []
        for parent, child in relations:
            arg1_id = f"{id}_cluster_{parent}"
            arg2_id = f"{id}_cluster_{child}"
            relations.append(
                {
                    "id": f"{id}_relation_{parent}_parent_of_{child}",
                    "type": "parent_of",
                    "arg1_id": arg1_id,
                    "arg2_id": arg2_id,
                    "normalized": [],
                }
            )

        bigbio_example["passages"] = passages
        bigbio_example["entities"] = entities
        bigbio_example["coreferences"] = coreferences
        bigbio_example["relations"] = relations

        return bigbio_example
