"""
Import keys for each schema
"""


class KBSchema:
    """
    Provides relevant keys for KB-style tasks

    :param keys: Mandatory keys
    :param opt_keys: at least 1 key must be present
    """

    def __init__(self):
        self.keys = ["id", "document_id", "passages"]
        self.opt_keys = ["entities", "events", "relations", "coreferences"]

        self.entities = ["id", "type", "text", "offsets", "normalized"]
        self.events = ["id", "type", "trigger", "arguments"]
        self.relations = ["id", "type", "arg1_id", "arg2_id", "normalized"]
        self.coreferences = ["id", "entity_ids"]


class QASchema:
    """
    Provides relevant keys for QA style tasks
    """

    def __init__(self):
        self.keys = [
            "id",
            "question_id",
            "document_id",
            "question",
            "type",
            "context",
            "answer",
        ]


class EntailmentSchema:
    """
    Provides relevant keys for Entailment style tasks
    """

    def __init__(self):
        self.keys = ["id", "premise", "hypothesis", "label"]


class Text2TextSchema:
    """
    Provides relevant keys for Entailment style tasks
    """

    def __init__(self):
        self.keys = [
            "id",
            "document_id",
            "text_1",
            "text_2",
            "text_1_name",
            "text_2_name",
        ]


class TextSchema:
    """
    Provides relevant keys for Entailment style tasks
    """

    def __init__(self):
        self.keys = ["id", "document_id", "text", "label"]


class PairsSchema:
    """
    Provides relevant keys for Entailment style tasks
    """

    def __init__(self):
        self.keys = ["id", "document_id", "text_1", "text_2", "label"]
