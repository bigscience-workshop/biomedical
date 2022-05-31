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
A large annotated corpus of patient eligibility criteria extracted from 1,000 interventional, Phase IV clinical
trials registered in ClinicalTrials.gov. This dataset includes 12,409 annotated eligibility criteria, represented
by 41,487 distinctive entities of 15 entity types and 25,017 relationships of 12 relationship types.
"""
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import datasets

import bigbio.utils.parsing as parsing
import bigbio.utils.schemas as schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.license import Licenses
from bigbio.utils.constants import Lang, Tasks

_LOCAL = False
_CITATION = """\
@article{kury2020chia,
  title={Chia, a large annotated corpus of clinical trial eligibility criteria},
  author={Kury, Fabr{\'\\i}cio and Butler, Alex and Yuan, Chi and Fu, Li-heng and Sun, Yingcheng and Liu,
          Hao and Sim, Ida and Carini, Simona and Weng, Chunhua},
  journal={Scientific data},
  volume={7},
  number={1},
  pages={1--11},
  year={2020},
  publisher={Nature Publishing Group}
}
"""

_DATASETNAME = "chia"

_DESCRIPTION = """\
A large annotated corpus of patient eligibility criteria extracted from 1,000 interventional, Phase IV clinical
trials registered in ClinicalTrials.gov. This dataset includes 12,409 annotated eligibility criteria, represented
by 41,487 distinctive entities of 15 entity types and 25,017 relationships of 12 relationship types.
"""

_HOMEPAGE = "https://github.com/WengLab-InformaticsResearch/CHIA"
_LICENSE = Licenses.CC_BY_4p0

_URLS = {
    _DATASETNAME: "https://figshare.com/ndownloader/files/21728850",
    _DATASETNAME + "_wo_scope": "https://figshare.com/ndownloader/files/21728853",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "2.0.0"
_BIGBIO_VERSION = "1.0.0"

# For further information see appendix of the publication
_DOMAIN_ENTITY_TYPES = [
    "Condition",
    "Device",
    "Drug",
    "Measurement",
    "Observation",
    "Person",
    "Procedure",
    "Visit",
]

# For further information see appendix of the publication
_FIELD_ENTITY_TYPES = [
    "Temporal",
    "Value",
]

# For further information see appendix of the publication
_CONSTRUCT_ENTITY_TYPES = [
    "Scope",  # Not part of the "without scope" schema / version
    "Negation",
    "Multiplier",
    "Qualifier",
    "Reference_point",
    "Mood",
]

_ALL_ENTITY_TYPES = _DOMAIN_ENTITY_TYPES + _FIELD_ENTITY_TYPES + _CONSTRUCT_ENTITY_TYPES

_RELATION_TYPES = [
    "AND",
    "OR",
    "SUBSUMES",
    "HAS_NEGATION",
    "HAS_MULTIPLIER",
    "HAS_QUALIFIER",
    "HAS_VALUE",
    "HAS_TEMPORAL",
    "HAS_INDEX",
    "HAS_MOOD",
    "HAS_CONTEXT ",
    "HAS_SCOPE",  # Not part of the "without scope" schema / version
]

_MAX_OFFSET_CORRECTION = 100


class ChiaDataset(datasets.GeneratorBasedBuilder):
    """
    A large annotated corpus of patient eligibility criteria extracted from 1,000 interventional,
    Phase IV clinical trials registered in ClinicalTrials.gov.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="chia_source",
            version=SOURCE_VERSION,
            description="Chia source schema",
            schema="source",
            subset_id="chia",
        ),
        BigBioConfig(
            name="chia_fixed_source",
            version=SOURCE_VERSION,
            description="Chia source schema (with fixed entity offsets)",
            schema="source",
            subset_id="chia_fixed",
        ),
        BigBioConfig(
            name="chia_without_scope_source",
            version=SOURCE_VERSION,
            description="Chia without scope source schema",
            schema="source",
            subset_id="chia_without_scope",
        ),
        BigBioConfig(
            name="chia_without_scope_fixed_source",
            version=SOURCE_VERSION,
            description="Chia without scope source schema (with fixed entity offsets)",
            schema="source",
            subset_id="chia_without_scope_fixed",
        ),
        BigBioConfig(
            name="chia_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Chia BigBio schema",
            schema="bigbio_kb",
            subset_id="chia",
        ),
    ]

    DEFAULT_CONFIG_NAME = "chia_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),  # NCT-ID from clinicialtrials.gov
                    "text": datasets.Value("string"),
                    "text_type": datasets.Value("string"),  # inclusion or exclusion (criteria)
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
        url_key = _DATASETNAME

        if self.config.subset_id.startswith("chia_without_scope"):
            url_key += "_wo_scope"

        urls = _URLS[url_key]
        data_dir = Path(dl_manager.download_and_extract(urls))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir},
            )
        ]

    def _generate_examples(self, data_dir: Path) -> Iterator[Tuple[str, Dict]]:
        if self.config.schema == "source":
            fix_offsets = "fixed" in self.config.subset_id

            for file in data_dir.iterdir():
                if not file.name.endswith(".txt"):
                    continue

                brat_example = parse_brat_file(file, [".ann"])
                source_example = self._to_source_example(file, brat_example, fix_offsets)
                yield source_example["id"], source_example

        elif self.config.schema == "bigbio_kb":
            for file in data_dir.iterdir():
                if not file.name.endswith(".txt"):
                    continue

                brat_example = parse_brat_file(file, [".ann"])
                source_example = self._to_source_example(file, brat_example, True)

                bigbio_example = {
                    "id": source_example["id"],
                    "document_id": source_example["document_id"],
                    "passages": [
                        {
                            "id": source_example["id"] + "_text",
                            "type": source_example["text_type"],
                            "text": [source_example["text"]],
                            "offsets": [[0, len(source_example["text"])]],
                        }
                    ],
                    "entities": source_example["entities"],
                    "relations": source_example["relations"],
                    "events": [],
                    "coreferences": [],
                }

                yield bigbio_example["id"], bigbio_example

    def _to_source_example(self, input_file: Path, brat_example: Dict, fix_offsets: bool) -> Dict:
        """
        Converts the generic brat example to the source schema format.
        """
        example_id = str(input_file.stem)
        document_id = example_id.split("_")[0]
        criteria_type = "inclusion" if "_inc" in input_file.stem else "exclusion"

        text = brat_example["text"]

        source_example = {
            "id": example_id,
            "document_id": document_id,
            "text_type": criteria_type,
            "text": text,
            "entities": [],
            "relations": [],
        }

        example_prefix = example_id + "_"
        entity_ids = {}

        for tb_annotation in brat_example["text_bound_annotations"]:
            if tb_annotation["type"].capitalize() not in _ALL_ENTITY_TYPES:
                continue

            entity_ann = tb_annotation.copy()
            entity_ann["id"] = example_prefix + entity_ann["id"]
            entity_ids[entity_ann["id"]] = True

            if fix_offsets:
                if len(entity_ann["offsets"]) > 1:
                    entity_ann["text"] = self._get_texts_for_multiple_offsets(text, entity_ann["offsets"])

                fixed_offsets = []
                fixed_texts = []
                for entity_text, offsets in zip(entity_ann["text"], entity_ann["offsets"]):
                    fixed_offset = self._fix_entity_offsets(text, entity_text, offsets)
                    fixed_offsets.append(fixed_offset)
                    fixed_texts.append(text[fixed_offset[0] : fixed_offset[1]])

                entity_ann["offsets"] = fixed_offsets
                entity_ann["text"] = fixed_texts

            entity_ann["normalized"] = []
            source_example["entities"].append(entity_ann)

        for base_rel_annotation in brat_example["relations"]:
            if base_rel_annotation["type"].upper() not in _RELATION_TYPES:
                continue

            head_id = example_prefix + base_rel_annotation["head"]["ref_id"]
            tail_id = example_prefix + base_rel_annotation["tail"]["ref_id"]

            if head_id not in entity_ids or tail_id not in entity_ids:
                continue

            relation = {
                "id": example_prefix + base_rel_annotation["id"],
                "type": base_rel_annotation["type"],
                "arg1_id": head_id,
                "arg2_id": tail_id,
                "normalized": [],
            }

            source_example["relations"].append(relation)

        relation_id = len(brat_example["relations"]) + 10
        for base_co_reference in brat_example["equivalences"]:
            ref_ids = base_co_reference["ref_ids"]
            for i, arg1 in enumerate(ref_ids[:-1]):
                for arg2 in ref_ids[i + 1 :]:
                    if arg1 not in entity_ids or arg2 not in entity_ids:
                        continue

                    or_relation = {
                        "id": example_prefix + f"R{relation_id}",
                        "type": "OR",
                        "arg1_id": example_prefix + arg1,
                        "arg2_id": example_prefix + arg2,
                        "normalized": [],
                    }

                    source_example["relations"].append(or_relation)
                    relation_id += 1

        return source_example

    def _fix_entity_offsets(self, doc_text: str, entity_text: str, given_offsets: List[int]) -> List[int]:
        """
        Fixes incorrect mention offsets by checking whether the given entity mention text can be
        found to the left or right of the given offsets by considering incrementally larger shifts.
        """
        left = given_offsets[0]
        right = given_offsets[1]

        # Some annotations contain whitespaces - we ignore them
        clean_entity_text = entity_text.strip()

        i = 0
        while i <= _MAX_OFFSET_CORRECTION:
            # Move mention window to the left
            if doc_text[left - i : right - i].strip() == clean_entity_text:
                return [left - i, left - i + len(clean_entity_text)]

            # Move mention window to the right
            elif doc_text[left + i : right + i].strip() == clean_entity_text:
                return [left + i, left + i + len(clean_entity_text)]

            i += 1

        # We can't find any better offsets
        return given_offsets

    def _get_texts_for_multiple_offsets(self, document_text: str, offsets: List[List[int]]) -> List[str]:
        """
        Extracts the single text span for a given list of offsets.
        """
        texts = []
        for offset in offsets:
            texts.append(document_text[offset[0] : offset[1]])
        return texts


def parse_brat_file(txt_file: Path, annotation_file_suffixes: List[str] = None) -> Dict:
    """
    Parse a brat file into the schema defined below.
    `txt_file` should be the path to the brat '.txt' file you want to parse, e.g. 'data/1234.txt'
    Assumes that the annotations are contained in one or more of the corresponding '.a1', '.a2' or '.ann' files,
    e.g. 'data/1234.ann' or 'data/1234.a1' and 'data/1234.a2'.

    Schema of the parse:
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
                            "trigger": datasets.Value(
                                "string"
                            ),  # refers to the text_bound_annotation of the trigger,
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
                            "resource_name": datasets.Value(
                                "string"
                            ),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value(
                                "string"
                            ),  # ID in the resource, e.g. 534366
                            "text": datasets.Value(
                                "string"
                            ),  # Human readable description/name of the entity, e.g. "Barack Obama"
                        }
                    ],
                },
            )
    """

    example = {}
    example["document_id"] = txt_file.with_suffix("").name
    with txt_file.open() as f:
        example["text"] = f.read()

    # If no specific suffixes of the to-be-read annotation files are given - take standard suffixes
    # for event extraction
    if annotation_file_suffixes is None:
        annotation_file_suffixes = [".a1", ".a2", ".ann"]

    if len(annotation_file_suffixes) == 0:
        raise AssertionError("At least one suffix for the to-be-read annotation files should be given!")

    ann_lines = []
    for suffix in annotation_file_suffixes:
        annotation_file = txt_file.with_suffix(suffix)
        if annotation_file.exists():
            with annotation_file.open() as f:
                ann_lines.extend(f.readlines())

    example["text_bound_annotations"] = []
    example["events"] = []
    example["relations"] = []
    example["equivalences"] = []
    example["attributes"] = []
    example["normalizations"] = []

    prev_tb_annotation = None

    for line in ann_lines:
        orig_line = line
        line = line.strip()
        if not line:
            continue

        # If an (entity) annotation spans multiple lines, this will result in multiple
        # lines also in the annotation file
        if "\t" not in line and prev_tb_annotation is not None:
            prev_tb_annotation["text"][0] += "\n" + orig_line[:-1]
            continue

        if line.startswith("T"):  # Text bound
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["text"] = [fields[2]]
            ann["type"] = fields[1].split()[0]
            ann["offsets"] = []
            span_str = parsing.remove_prefix(fields[1], (ann["type"] + " "))
            for span in span_str.split(";"):
                start, end = span.split()
                ann["offsets"].append([int(start), int(end)])

            example["text_bound_annotations"].append(ann)
            prev_tb_annotation = ann

        elif line.startswith("E"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]

            ann["type"], ann["trigger"] = fields[1].split()[0].split(":")

            ann["arguments"] = []
            for role_ref_id in fields[1].split()[1:]:
                argument = {
                    "role": (role_ref_id.split(":"))[0],
                    "ref_id": (role_ref_id.split(":"))[1],
                }
                ann["arguments"].append(argument)

            example["events"].append(ann)
            prev_tb_annotation = None

        elif line.startswith("R"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["type"] = fields[1].split()[0]

            ann["head"] = {
                "role": fields[1].split()[1].split(":")[0],
                "ref_id": fields[1].split()[1].split(":")[1],
            }
            ann["tail"] = {
                "role": fields[1].split()[2].split(":")[0],
                "ref_id": fields[1].split()[2].split(":")[1],
            }

            example["relations"].append(ann)
            prev_tb_annotation = None

        # '*' seems to be the legacy way to mark equivalences,
        # but I couldn't find any info on the current way
        # this might have to be adapted dependent on the brat version
        # of the annotation
        elif line.startswith("*"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["ref_ids"] = fields[1].split()[1:]

            example["equivalences"].append(ann)
            prev_tb_annotation = None

        elif line.startswith("A") or line.startswith("M"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]

            info = fields[1].split()
            ann["type"] = info[0]
            ann["ref_id"] = info[1]

            if len(info) > 2:
                ann["value"] = info[2]
            else:
                ann["value"] = ""

            example["attributes"].append(ann)
            prev_tb_annotation = None

        elif line.startswith("N"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["text"] = fields[2]

            info = fields[1].split()

            ann["type"] = info[0]
            ann["ref_id"] = info[1]
            ann["resource_name"] = info[2].split(":")[0]
            ann["cuid"] = info[2].split(":")[1]

            example["normalizations"].append(ann)
            prev_tb_annotation = None

    return example
