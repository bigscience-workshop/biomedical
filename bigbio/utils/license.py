#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define LICENCE classes to normalize _LICENSE tag of dataloader
"""

from dataclasses import dataclass
from typing import Optional


class LicenseMixin:
    """
    Base class from which all licenses inherit
    """

    def __repr__(self):
        """
        Override to get a formatted full description of the license
        """
        text = f"{self.name}"

        if hasattr(self, "type") and getattr(self, "type", None) is not None:
            text += f" - {self.type} -"

        if hasattr(self, "version") and getattr(self, "version", None) is not None:
            text += f" v{self.version}"

        if hasattr(self, "link") and getattr(self, "link", None) is not None:
            text += f" ({self.link})"

        if (
            hasattr(self, "description")
            and getattr(self, "description", None) is not None
        ):
            text += f": {self.description}"

        return text


@dataclass
class CreativeCommons(LicenseMixin):
    """
    All CC.
    """

    type: str
    version: int
    name: str = "CC"


@dataclass
class Apache(LicenseMixin):
    """
    All Apache.
    """

    version: int
    name: str = "Apache"


@dataclass
class GPL(LicenseMixin):
    """
    All GPL.
    """

    version: int
    name: str = "GNU General Public License"


@dataclass
class MIT(LicenseMixin):
    """
    All MIT.
    """

    name: str = "MIT"


@dataclass
class DUA(LicenseMixin):
    """
    This is for all datasets which can be freely downloaded but no licese is specified
    """

    name = "DUA"
    text = "Data User Agreement"
    type: Optional[str] = None


@dataclass
class PubliclyAvailable(LicenseMixin):
    """
    This class is for all datasets which can be freely downloaded but no licese is specified
    """

    text = "No license specified."
    name = "Publicly available"


@dataclass
class Custom(LicenseMixin):
    """
    This class is for custom licesense.
    It must contain the text describing the license and optionally a link as source
    """

    text = "CUSTOM"
    description: str
    link: Optional[str] = None
