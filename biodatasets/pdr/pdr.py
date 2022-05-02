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
The corpus of plant-disease relation consists of plants and diseases and their relation to PubMed abstract.
The corpus consists of about 2400 plant and disease entities and 300 annotated relations from 179 abstracts.

The big-bio and source version of this script are made by merging the 2 provided annotations on locations they intersected.
Both annotations (1, 2) are provided as separate source schemas.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import datasets

import bigbio.utils.parsing as parsing
import bigbio.utils.schemas as schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_CITATION = """\
@article{kim2019corpus,
  title={A corpus of plant--disease relations in the biomedical domain},
  author={Kim, Baeksoo and Choi, Wonjun and Lee, Hyunju},
  journal={PLoS One},
  volume={14},
  number={8},
  pages={e0221582},
  year={2019},
  publisher={Public Library of Science San Francisco, CA USA}
}
"""

_DATASETNAME = "pdr"

_DESCRIPTION = """
The corpus of plant-disease relation consists of plants and diseases and their relation to PubMed abstract.
The corpus consists of about 2400 plant and disease entities and 300 annotated relations from 179 abstracts.
"""

_HOMEPAGE = "http://gcancer.org/pdr/"
_LICENSE = ""

_URLS = {_DATASETNAME: "http://gcancer.org/pdr/Plant-Disease_Corpus.tar.gz"}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.RELATION_EXTRACTION,
    Tasks.EVENT_EXTRACTION,
    Tasks.COREFERENCE_RESOLUTION,
]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_ENTITY_TYPES = ["Plant", "Disease"]


class PDRDataset(datasets.GeneratorBasedBuilder):
    """The corpus of plant-disease relation consists of plants and diseases and their relation to PubMed abstract"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pdr_annotator1_source",
            version=SOURCE_VERSION,
            description="PDR annotator 1 source schema",
            schema="source",
            subset_id="pdr_annotator1",
        ),
        BigBioConfig(
            name="pdr_annotator2_source",
            version=SOURCE_VERSION,
            description="PDR annotator 2 source schema",
            schema="source",
            subset_id="pdr_annotator2",
        ),
        BigBioConfig(
            name="pdr_source",
            version=SOURCE_VERSION,
            description="PDR source schema",
            schema="source",
            subset_id="pdr",
        ),
        BigBioConfig(
            name="pdr_bigbio_kb",
            version=BIGBIO_VERSION,
            description="PDR BigBio schema",
            schema="bigbio_kb",
            subset_id="pdr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pdr_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "events": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            # refers to the text_bound_annotation of the trigger
                            "trigger": {
                                "text": datasets.Sequence(datasets.Value("string")),
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                            },
                            "arguments": [
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "coreferences": [
                        {
                            "id": datasets.Value("string"),
                            "entity_ids": datasets.Sequence(datasets.Value("string")),
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
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls))
        data_dir = data_dir / "Plant-Disease_Corpus"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir},
            )
        ]

    def _generate_examples(self, data_dir: Path) -> Iterator[Tuple[str, Dict]]:
        if self.config.schema == "source":
            for file in data_dir.iterdir():
                if not str(file).endswith(".txt"):
                    continue

                if self.config.subset_id == "pdr_annotator1":
                    # Provide annotations of annotator 1
                    example = parsing.parse_brat_file(file, [".ann"])
                    example = parsing.brat_parse_to_bigbio_kb(example, _ENTITY_TYPES)

                elif self.config.subset_id == "pdr_annotator2":
                    # Provide annotations of annotator 2
                    example = parsing.parse_brat_file(file, [".ann2"])
                    example = parsing.brat_parse_to_bigbio_kb(example, _ENTITY_TYPES)

                elif self.config.subset_id == "pdr":
                    # Provide merged version of annotator 1 and 2
                    annotator1 = parsing.parse_brat_file(file, [".ann"])
                    annotator1 = parsing.brat_parse_to_bigbio_kb(annotator1, _ENTITY_TYPES)

                    annotator2 = parsing.parse_brat_file(file, [".ann2"])
                    annotator2 = parsing.brat_parse_to_bigbio_kb(annotator2, _ENTITY_TYPES)

                    example = self._merge_annotations_by_intersection(file, annotator1, annotator2)

                example["text"] = example["passages"][0]["text"][0]
                example.pop("id", None)
                example.pop("passages", None)

                yield example["document_id"], example

        elif self.config.schema == "bigbio_kb":
            for file in data_dir.iterdir():
                if not str(file).endswith(".txt"):
                    continue

                annotator1 = parsing.parse_brat_file(file, [".ann"])
                annotator1 = parsing.brat_parse_to_bigbio_kb(annotator1, _ENTITY_TYPES)

                annotator2 = parsing.parse_brat_file(file, [".ann2"])
                annotator2 = parsing.brat_parse_to_bigbio_kb(annotator2, _ENTITY_TYPES)

                merged_annotation = self._merge_annotations_by_intersection(file, annotator1, annotator2)
                merged_annotation["id"] = merged_annotation["document_id"]

                yield merged_annotation["id"], merged_annotation

    def _merge_annotations_by_intersection(self, file: Path, example_ann1: Dict, example_ann2: Dict) -> Dict:
        """
        Merges the two given examples by only keeping annotations on which both annotators agree.
        """
        id_prefix = str(file.stem) + "_"

        # Mapping entity identifiers from annotator 1 / 2 to merged entity ids
        a1_entity_to_merged_entity = {}
        a2_entity_to_merged_entity = {}
        merged_entities = []

        # 1. Find all common entities, i.e. both annotators agree on same type and their offsets overlap
        entity_id = 1
        for entity1 in example_ann1["entities"]:
            for entity2 in example_ann2["entities"]:
                if self._overlaps(entity1, entity2) and entity1["type"] == entity2["type"]:
                    text_entity1 = "".join(entity1["text"])
                    text_entity2 = "".join(entity2["text"])

                    longer_entity = entity1 if len(text_entity1) > len(text_entity2) else entity2
                    merged_entity_id = id_prefix + f"E{entity_id}"
                    entity_id += 1

                    merged_entity = longer_entity.copy()
                    merged_entity["id"] = merged_entity_id
                    merged_entity["normalized"] = []
                    merged_entities.append(merged_entity)

                    a1_entity_to_merged_entity[entity1["id"]] = merged_entity_id
                    a2_entity_to_merged_entity[entity2["id"]] = merged_entity_id
                    break

        # Find all relations the two annotators agree on
        relations_ann1 = self._map_relations(example_ann1, a1_entity_to_merged_entity)
        relations_ann2 = self._map_relations(example_ann2, a2_entity_to_merged_entity)
        relations = []
        relation_id = 1

        for rel_type, relations_1 in relations_ann1.items():
            relations_2 = relations_ann2[rel_type]

            for relation_pair_1 in relations_1:
                for relation_pair_2 in relations_2:
                    if relation_pair_1 == relation_pair_2:
                        relations.append(
                            {
                                "id": id_prefix + f"R{relation_id}",
                                "type": rel_type,
                                "arg1_id": relation_pair_1[0],
                                "arg2_id": relation_pair_1[1],
                                "normalized": [],
                            }
                        )
                        relation_id += 1
                        break

        # Find all events the two annotators agree on
        events_ann1 = self._map_events(example_ann1, a1_entity_to_merged_entity)
        events_ann2 = self._map_events(example_ann2, a2_entity_to_merged_entity)
        events = []
        event_id = 1

        for event_type, events_1 in events_ann1.items():
            events_2 = events_ann2[event_type]

            for (trigger1, theme1, cause1) in events_1:
                for (trigger2, theme2, cause2) in events_2:
                    if theme1 == theme2 and cause1 == cause2 and self._overlaps(trigger1, trigger2):
                        trigger1_text = "".join(trigger1["text"])
                        trigger2_text = "".join(trigger2["text"])

                        longer_trigger = trigger1 if len(trigger1_text) >= len(trigger2_text) else trigger2
                        events.append(
                            {
                                "id": id_prefix + f"T{event_id}",
                                "type": event_type,
                                "trigger": longer_trigger,
                                "arguments": [
                                    {"role": "Theme", "ref_id": theme1},
                                    {"role": "Cause", "ref_id": cause1},
                                ],
                            }
                        )
                        event_id += 1
                        break

        # Find all coreferences the annotators agree on
        coferences_ann1 = self._map_coreferences(example_ann1, a1_entity_to_merged_entity)
        coferences_ann2 = self._map_coreferences(example_ann2, a2_entity_to_merged_entity)
        coreferences = []
        coreference_id = 1

        for _, entity_ids1 in coferences_ann1.items():
            for _, entity_ids2 in coferences_ann2.items():
                if entity_ids1.intersection(entity_ids2) == entity_ids1.union(entity_ids2):
                    coreferences.append({"id": id_prefix + f"CO{coreference_id}", "entity_ids": list(entity_ids1)})
                    coreference_id += 1

        merged_example = example_ann1.copy()
        merged_example["entities"] = merged_entities
        merged_example["relations"] = relations
        merged_example["events"] = events
        merged_example["coreferences"] = coreferences

        return merged_example

    def _map_relations(self, example: Dict, entity_id_mapping: Dict) -> Dict:
        """
        Maps the all relations of the given example to their merged entity identifiers
        (if existent)
        """
        relation_map = defaultdict(list)

        for relation in example["relations"]:
            arg1_id = relation["arg1_id"]
            arg2_id = relation["arg2_id"]

            # Are both entities also in the merged version?
            if arg1_id not in entity_id_mapping or arg2_id not in entity_id_mapping:
                continue

            com_arg1_id = entity_id_mapping[arg1_id]
            com_arg2_id = entity_id_mapping[arg2_id]

            relation_map[relation["type"]].append((com_arg1_id, com_arg2_id))

        return relation_map

    def _map_events(self, example: Dict, entity_id_mapping: Dict) -> Dict:
        """
        Maps the all events of the given example to their merged entity identifiers
        (if existent)
        """
        event_map = defaultdict(list)

        for event in example["events"]:
            theme_id = self._get_event_argument(event, "Theme")
            cause_id = self._get_event_argument(event, "Cause")

            if theme_id not in entity_id_mapping or cause_id not in entity_id_mapping:
                continue

            common_theme_id = entity_id_mapping[theme_id]
            common_cause_id = entity_id_mapping[cause_id]

            event_map[event["type"]].append((event["trigger"], common_theme_id, common_cause_id))

        return event_map

    def _map_coreferences(self, annotation: Dict, entity_mapping: Dict) -> Dict:
        """
        Maps the all coreferences of the given example to their merged entity identifiers
        (if existent)
        """
        id_to_corefs = defaultdict(set)
        for coreference in annotation["coreferences"]:
            entity_ids = set([entity_mapping[id] for id in coreference["entity_ids"] if id in entity_mapping])

            # Are both id's also in the merged version?
            if len(entity_ids) > 1:
                id_to_corefs[coreference["id"]] = entity_ids

        return id_to_corefs

    def _overlaps(self, annotation1: Dict, annotation2: Dict) -> bool:
        """
        Checks whether the offsets of the two given annotations overlap.
        """
        for (start1, end1) in annotation1["offsets"]:
            for (start2, end2) in annotation2["offsets"]:
                if (start2 <= start1 <= end2) or (start2 <= end1 <= end2):
                    return True

        return False

    def _get_event_argument(self, event: Dict, role: str) -> Optional[str]:
        """
        Returns the argument with the given role from the given event annotation.
        """
        for argument in event["arguments"]:
            if argument["role"] == role:
                return argument["ref_id"]

        return None
