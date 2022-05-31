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

    Args:
        name: License title
        text: Accompanying information of the license
        link: URL to License
        version: Current version of license
        provenance: Organization providing authorization, if possible
    """

    name: Optional[str] = None
    text: Optional[str] = None
    link: Optional[str] = None
    version: Optional[float] = None
    provenance: Optional[str] = None

    @property
    def is_share_alike(self):
        """
        Is Share-alike?
        """
        # NOTE: leave here has an example of license properties
        raise NotImplementedError()


@dataclass
class CustomLicense(License):
    """
    This class is for custom licenses.
    It must contain the text of the license.
    Optionally its version and a link to the license webpage.
    """

    def __post_init__(self):
        if self.name is None:
            self.name = "Custom license"

        if self.text is None or self.link is None:
            raise ValueError(
                "A `CustomLicense` must provide (a) the license text or (b) the license link!"
            )


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
Licenses = Enum("Licenses", {k: License(name=v) for k, v in _LICENSES.items()})
