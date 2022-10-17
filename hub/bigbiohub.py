from dataclasses import dataclass
from enum import Enum
import datasets
from types import SimpleNamespace


BigBioValues = SimpleNamespace(NULL="<BB_NULL_STR>")


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None


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
