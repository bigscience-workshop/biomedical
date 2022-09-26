import json
from collections import defaultdict
from enum import Enum
from types import SimpleNamespace

from dataclasses import dataclass
import datasets

from licenses import License
from licenses import Licenses


BigBioValues = SimpleNamespace(NULL="<BB_NULL_STR>")


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None


# shamelessly compied from:
# https://github.com/huggingface/datasets/blob/master/src/datasets/utils/metadata.py
langs_json = json.load(open("languages.json", "r"))
langs_dict = {k.replace("-", "_").upper(): v for k, v in langs_json.items()}
Lang = Enum("Lang", langs_dict)


METADATA: dict = {
    "_LOCAL": bool,
    "_LANGUAGES": Lang,
    "_PUBMED": bool,
    "_LICENSE": License,
    "_DISPLAYNAME": str,
}


class Tasks(Enum):
    NAMED_ENTITY_RECOGNITION = "NER"
    NAMED_ENTITY_DISAMBIGUATION = "NED"
    EVENT_EXTRACTION = "EE"
    RELATION_EXTRACTION = "RE"
    COREFERENCE_RESOLUTION = "COREF"
    QUESTION_ANSWERING = "QA"
    TEXTUAL_ENTAILMENT = "TE"
    SEMANTIC_SIMILARITY = "STS"
    TEXT_PAIRS_CLASSIFICATION = "TXT2CLASS"
    PARAPHRASING = "PARA"
    TRANSLATION = "TRANSL"
    SUMMARIZATION = "SUM"
    TEXT_CLASSIFICATION = "TXTCLASS"


TASK_TO_SCHEMA = {
    Tasks.NAMED_ENTITY_RECOGNITION: "KB",
    Tasks.NAMED_ENTITY_DISAMBIGUATION: "KB",
    Tasks.EVENT_EXTRACTION: "KB",
    Tasks.RELATION_EXTRACTION: "KB",
    Tasks.COREFERENCE_RESOLUTION: "KB",
    Tasks.QUESTION_ANSWERING: "QA",
    Tasks.TEXTUAL_ENTAILMENT: "TE",
    Tasks.SEMANTIC_SIMILARITY: "PAIRS",
    Tasks.TEXT_PAIRS_CLASSIFICATION: "PAIRS",
    Tasks.PARAPHRASING: "T2T",
    Tasks.TRANSLATION: "T2T",
    Tasks.SUMMARIZATION: "T2T",
    Tasks.TEXT_CLASSIFICATION: "TEXT",
}

SCHEMA_TO_TASKS = defaultdict(set)
for task, schema in TASK_TO_SCHEMA.items():
    SCHEMA_TO_TASKS[schema].add(task)
SCHEMA_TO_TASKS = dict(SCHEMA_TO_TASKS)

VALID_TASKS = set(TASK_TO_SCHEMA.keys())
VALID_SCHEMAS = set(TASK_TO_SCHEMA.values())


entailment_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "premise": datasets.Value("string"),
        "hypothesis": datasets.Value("string"),
        "label": datasets.Value("string"),
    }
)

pairs_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "text_1": datasets.Value("string"),
        "text_2": datasets.Value("string"),
        "label": datasets.Value("string"),
    }
)

qa_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question_id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "type": datasets.Value("string"),
        "choices": [datasets.Value("string")],
        "context": datasets.Value("string"),
        "answer": datasets.Sequence(datasets.Value("string")),
    }
)

text_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "text": datasets.Value("string"),
        "labels": [datasets.Value("string")],
    }
)

text2text_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "text_1": datasets.Value("string"),
        "text_2": datasets.Value("string"),
        "text_1_name": datasets.Value("string"),
        "text_2_name": datasets.Value("string"),
    }
)

kb_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "passages": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "text": datasets.Sequence(datasets.Value("string")),
                "offsets": datasets.Sequence([datasets.Value("int32")]),
            }
        ],
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

SCHEMA_TO_FEATURES = {
    "KB": kb_features,
    "QA": qa_features,
    "TE": entailment_features,
    "T2T": text2text_features,
    "TEXT": text_features,
    "PAIRS": pairs_features,
}
