from enum import Enum
from types import SimpleNamespace

from bigbio.utils.schemas import (
    entailment_features,
    kb_features,
    pairs_features,
    qa_features,
    text2text_features,
    text_features,
)

BigBioValues = SimpleNamespace(NULL="<BB_NULL_STR>")

class Tasks(Enum):
    NAMED_ENTITY_RECOGNITION = "NER"
    NAMED_ENTITY_DISAMBIGUATION = "NED"
    EVENT_EXTRACTION = "EE"
    RELATION_EXTRACTION = "RE"
    COREFERENCE_RESOLUTION = "COREF"

    QUESTION_ANSWERING = "QA"

    TEXTUAL_ENTAILMENT = "TE"

    SEMANTIC_SIMILARITY = "STS"

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
    Tasks.PARAPHRASING: "T2T",
    Tasks.TRANSLATION: "T2T",
    Tasks.SUMMARIZATION: "T2T",
    Tasks.TEXT_CLASSIFICATION: "TEXT",
}

VALID_TASKS = set(TASK_TO_SCHEMA.keys())
VALID_SCHEMAS = set(TASK_TO_SCHEMA.values())

SCHEMA_TO_FEATURES = {
    "KB": kb_features,
    "QA": qa_features,
    "TE": entailment_features,
    "T2T": text2text_features,
    "TEXT": text_features,
    "PAIRS": pairs_features,
}
