#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to create dummy examples on the fly.
"""

from collections import UserDict
from typing import Optional

_DOCUMENT_ID = 0
_TYPE = "type"
_PASSAGE_TEXT = ["This is a dummy passage."]
_PASSAGE_OFFSETS = [[0, 24]]
_ENTITY_TEXT_1 = "is"
_ENTITY_TEXT_2 = "dummy"
_ENTITY_OFFSET_1 = [5, 7]
_ENTITY_OFFSET_2 = [10, 15]
_ENTITY_NORMALIZED = [{"db_name": "DB", "db_id": "-1"}]
_RELATION_NORMALIZED = _ENTITY_NORMALIZED
_EVENT_ARGUMENTS = [{"role": "ROLE", "ref_id": "-1"}]


class DummyBase(UserDict):
    """
    Dummy dict of features which allows to override specific keys
    """

    def __init__(
        self,
        uid: str,
        override: Optional[dict] = None,
        remove: Optional[list] = None,
    ):
        super().__init__(**self.data)

        self["id"] = uid

        if override is not None:

            for key, value in override.items():
                self[key] = value

        if remove is not None:
            for key in remove:
                self.pop(key)


class DummyPassage(DummyBase):
    """
    Dummy `passage` feature of KB schema
    """

    data = {"type": _TYPE, "text": _PASSAGE_TEXT, "offsets": _PASSAGE_OFFSETS}

    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DummyEntity(DummyBase):
    """
    Dummy `entity` feature of KB schema
    """

    data = {
        "text": [_ENTITY_TEXT_1, _ENTITY_TEXT_2],
        "type": _TYPE,
        "offsets": [_ENTITY_OFFSET_1, _ENTITY_OFFSET_2],
        "normalized": _ENTITY_NORMALIZED,
    }

    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DummyRelation(DummyBase):
    """
    Dummy `relation` feature of KB schema
    """

    data = {"type": _TYPE, "normalized": _RELATION_NORMALIZED}

    def __init__(self, arg1_id: str, arg2_id: str, *args, **kwargs):
        self["arg1_id"] = arg1_id
        self["arg2_id"] = arg2_id
        super().__init__(*args, **kwargs)


class DummyCoreference(DummyBase):
    """
    Dummy `coreference` feature of KB schema
    """

    data: dict = {}

    def __init__(self, entity_ids: list[str], *args, **kwargs):
        self["entity_ids"] = entity_ids
        super().__init__(*args, **kwargs)


class DummyEvent(DummyBase):
    """
    Dummy `event` feature of KB schema
    """

    data = {
        "type": _TYPE,
        "trigger": {"text": [_ENTITY_TEXT_1], "offsets": [_ENTITY_OFFSET_1]},
        "arguments": _EVENT_ARGUMENTS,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DummyKBExample(DummyBase):
    """
    Dummy working KB Example with all features
    """

    data = {
        "passages": [DummyPassage(uid=1)],
        "entities": [DummyEntity(uid=2), DummyEntity(uid=3)],
        "relations": [DummyRelation(uid=4, arg1_id=2, arg2_id=3)],
        "coreferences": [DummyCoreference(uid=5, entity_ids=[2, 3])],
        "events": [DummyEvent(uid=6)],
    }

    def __init__(self, document_id: int, *args, **kwargs):
        self["document_id"] = document_id
        super().__init__(*args, **kwargs)
