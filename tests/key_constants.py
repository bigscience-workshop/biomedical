"""
Import keys for each schema
"""

class KBSchema:
    """
    Instantiate relevant keys for KB-style tasks
    """
    def __init__(self):
        self.keys = ["id", "document_id", "passages"]
        self.opt_keys = ["entities", "events", "relations", "coreferences"]

        self.entities = ["id", "type", "text", "offsets", "normalized"]
        self.events = ["id", "type", "trigger", "arguments"]
        self.relations = ["id", "type", "arg1_id", "arg2_id", "normalized"]
        self.coreferences = ["id", "entity_ids"]