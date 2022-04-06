from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def remove_prefix(a: str, prefix: str) -> str:
    if a.startswith(prefix):
        a = a[len(prefix) :]
    return a


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
    for line in ann_lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("T"):  # Text bound
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["text"] = [fields[2]]
            ann["type"] = fields[1].split()[0]
            ann["offsets"] = []
            span_str = remove_prefix(fields[1], (ann["type"] + " "))
            for span in span_str.split(";"):
                start, end = span.split()
                ann["offsets"].append([int(start), int(end)])

            example["text_bound_annotations"].append(ann)

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

    return example


def brat_parse_to_bigbio_kb(brat_parse: Dict, entity_types: Iterable[str]) -> Dict:
    """
    Transform a brat parse (conforming to the standard brat schema) obtained with
    `parse_brat_file` into a dictionary conforming to the `bigbio-kb` schema (as defined in ../schemas/kb.py)

    :param brat_parse:
    :param entity_types: Entity types of the dataset. This should include all types of `T` annotations that are not
                         event triggers and will be different in different datasets.
    """

    unified_example = {}

    # Prefix all ids with document id to ensure global uniqueness,
    # because brat ids are only unique within their document
    id_prefix = brat_parse["document_id"] + "_"

    # identical
    unified_example["document_id"] = brat_parse["document_id"]
    unified_example["passages"] = [
        {
            "id": id_prefix + "_text",
            "type": "abstract",
            "text": [brat_parse["text"]],
            "offsets": [[0, len(brat_parse["text"])]],
        }
    ]

    # get normalizations
    ref_id_to_normalizations = defaultdict(list)
    for normalization in brat_parse["normalizations"]:
        ref_id_to_normalizations[normalization["ref_id"]].append(
            {
                "db_name": normalization["resource_name"],
                "db_id": normalization["cuid"],
            }
        )

    # separate entities and event triggers
    unified_example["entities"] = []
    id_to_event_trigger = {}
    for ann in brat_parse["text_bound_annotations"]:
        if ann["type"] in entity_types:
            entity_ann = ann.copy()
            entity_ann["id"] = id_prefix + entity_ann["id"]
            entity_ann["normalized"] = ref_id_to_normalizations[ann["id"]]
            unified_example["entities"].append(entity_ann)
        else:
            id_to_event_trigger[ann["id"]] = ann

    unified_example["events"] = []
    for event in brat_parse["events"]:
        event = event.copy()
        event["id"] = id_prefix + event["id"]
        trigger = id_to_event_trigger[event["trigger"]]
        event["trigger"] = {
            "text": trigger["text"].copy(),
            "offsets": trigger["offsets"].copy(),
        }
        for argument in event["arguments"]:
            argument["ref_id"] = id_prefix + argument["ref_id"]

        unified_example["events"].append(event)

    # massage relations
    unified_example["relations"] = []
    for ann in brat_parse["relations"]:
        unified_example["relations"].append(
            {
                "arg1_id": id_prefix + ann["head"]["ref_id"],
                "arg2_id": id_prefix + ann["tail"]["ref_id"],
                "id": id_prefix + ann["id"],
                "type": ann["type"],
                "normalized": [],
            }
        )

    # get coreferences
    unified_example["coreferences"] = []
    for i, ann in enumerate(brat_parse["equivalences"], start=1):
        is_entity_cluster = True
        for ref_id in ann["ref_ids"]:
            if not ref_id.startswith("T"):  # not textbound -> no entity
                is_entity_cluster = False
            elif ref_id in id_to_event_trigger:  # event trigger -> no entity
                is_entity_cluster = False
        if is_entity_cluster:
            entity_ids = [id_prefix + i for i in ann["ref_ids"]]
            unified_example["coreferences"].append({"id": id_prefix + str(i), "entity_ids": entity_ids})

    unified_example["n_ary_relations"] = []

    return unified_example
