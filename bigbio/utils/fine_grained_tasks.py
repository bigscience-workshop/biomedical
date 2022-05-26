#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-grained labeling of biomedical tasks
"""
import importlib.resources as pkg_resources
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from bigbio.utils import resources

@dataclass
class FineGrainedTask:
    """Baseclass for fine-grained tagging of tasks.

    Args:
        name: type of fine-grained task
        text: short description of task
    """

    name: Optional[str] = None
    text: Optional[str] = None


# Borrowed from the licensing dataclass
def _get_variable_name(k: str) -> str:

    return k.replace("-", "_").upper().replace(".", "p").replace("+", "plus")


def load_finegraintasks():
    """
    Load all possible fine-grained tasks from JSON file
    """

    # shamelessly compied from:
    # https://github.com/huggingface/datasets/blob/master/src/datasets/utils/metadata.py
    fgtasks = {
        _get_variable_name(k): v
        for k, v in json.loads(
            pkg_resources.read_text(resources, "fine_grained_tasks.json")
        ).items()
    }

    return fgtasks


_FGTasks = load_finegraintasks()
FineGrainedTasks = Enum(
    "FineGrainedTasks", {k: FineGrainedTask(name=v) for k, v in _FGTasks.items()}, type=FineGrainedTask
)