from enum import Enum


class BigBioValues(Enum):
    MISSING = "<BB-MISSING-VALUE>"  # to represent data that is not present in a dataset
    NULL = "<BB-NULL-VALUE">        # to represent data that is present in a dataset but has null meaning


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
