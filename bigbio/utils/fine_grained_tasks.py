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
    """