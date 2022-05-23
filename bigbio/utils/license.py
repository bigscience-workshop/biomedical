#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
License objects.
"""

import importlib.resources as pkg_resources
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from bigbio.utils import resources


@dataclass
class License:
    """
    Base class from which all licenses inherit
    """

    name: Optional[str] = None
    text: Optional[str] = None
    link: Optional[str] = None
    version: Optional[float] = None

    @property
    def is_share_alike(self):
        """
        Is Share-alike?
        """

        return False

    @property
    def display(self) -> str:
        """
        Get formatted name
        """

        string = f"{self.name}"

        if hasattr(self, "version") and getattr(self, "version", None) is not None:
            string += f" - v{self.version}"

        if hasattr(self, "link") and getattr(self, "link", None) is not None:
            string += f" ({self.link})"

        if hasattr(self, "text") and getattr(self, "text", None) is not None:
            string += f": {self.text}"

        return string

    def __repr__(self):
        """
        Override to get a formatted full description of the license
        """
        return self.display

    def __str__(self):
        """
        Override to get a formatted full description of the license
        """
        return self.display


@dataclass
class CustomLicense(License):
    """
    This class is for custom licenses.
    It must contain the text of the license.
    Optionally its version and a link to the license webpage.
    """

    def __post_init__(self):
        if self.text is None:
            raise ValueError(
                "A `CustomLicense` must provide at least the license text!"
            )
        if self.name is None:
            super().__setattr__("name", "CUSTOM")


def _get_variable_name(k: str) -> str:

    return k.replace("-", "_").upper().replace(".", "p").replace("+", "plus")


def load_licenses():
    """
    Load all licenses from JSON file.
    Amend names to be valid variable names
    """

    # shamelessly compied from:
    # https://github.com/huggingface/datasets/blob/master/src/datasets/utils/metadata.py
    licenses = {
        _get_variable_name(k): v
        for k, v in json.loads(
            pkg_resources.read_text(resources, "licenses.json")
        ).items()
    }

    licenses["ZERO_BSD"] = licenses.pop("0BSD")

    licenses.update(
        {"DUA": "Data User Agreement", "EXTERNAL_DUA": "External Data User Agreement"}
    )

    return licenses


_LICENSES = load_licenses()
Licenses = Enum(
    "Licenses", {k: License(name=v) for k, v in _LICENSES.items()}, type=License
)
