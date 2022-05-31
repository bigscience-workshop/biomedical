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
    version: Optional[str] = None
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


_GENIA_PROJECT_LICENSE_TEXT = """
GENIA Project License for Annotated Corpora

1. Copyright of abstracts

Any abstracts contained in this corpus are from PubMed(R), a database
of the U.S. National Library of Medicine (NLM).

NLM data are produced by a U.S. Government agency and include works of
the United States Government that are not protected by U.S. copyright
law but may be protected by non-US copyright law, as well as abstracts
originating from publications that may be protected by U.S. copyright
law.

NLM assumes no responsibility or liability associated with use of
copyrighted material, including transmitting, reproducing,
redistributing, or making commercial use of the data. NLM does not
provide legal advice regarding copyright, fair use, or other aspects
of intellectual property rights. Persons contemplating any type of
transmission or reproduction of copyrighted material such as abstracts
are advised to consult legal counsel.

2. Copyright of full texts

Any full texts contained in this corpus are from the PMC Open Access
Subset of PubMed Central (PMC), the U.S. National Institutes of Health
(NIH) free digital archive of biomedical and life sciences journal
literature.

Articles in the PMC Open Access Subset are protected by copyright, but
are made available under a Creative Commons or similar license that
generally allows more liberal redistribution and reuse than a
traditional copyrighted work. Please refer to the license of each
article for specific license terms.

3. Copyright of annotations

The copyrights of annotations created in the GENIA Project of Tsujii
Laboratory, University of Tokyo, belong in their entirety to the GENIA
Project.

4. Licence terms

Use and distribution of abstracts drawn from PubMed is subject to the
PubMed(R) license terms as stated in Clause 1.

Use and distribution of full texts is subject to the license terms
applying to each publication.

Annotations created by the GENIA Project are licensed under the
Creative Commons Attribution 3.0 Unported License. To view a copy of
this license, visit http://creativecommons.org/licenses/by/3.0/ or
send a letter to Creative Commons, 444 Castro Street, Suite 900,
Mountain View, California, 94041, USA.

Annotations created by the GENIA Project must be attributed as
detailed in Clause 5.

5. Attribution

The GENIA Project was founded and led by prof. Jun'ichi Tsujii and
the project and its annotation efforts have been coordinated in part
by Nigel Collier, Yuka Tateisi, Sang-Zoo Lee, Tomoko Ohta, Jin-Dong
Kim, and Sampo Pyysalo.

For a complete list of the GENIA Project members and contributors,
please refer to http://www.geniaproject.org.

The GENIA Project has been supported by Grant-in-Aid for Scientific
Research on Priority Area "Genome Information Science" (MEXT, Japan),
Grant-in-Aid for Scientific Research on Priority Area "Systems
Genomics" (MEXT, Japan), Core Research for Evolutional Science &
Technology (CREST) "Information Mobility Project" (JST, Japan),
Solution Oriented Research for Science and Technology (SORST) (JST,
Japan), Genome Network Project (MEXT, Japan) and Grant-in-Aid for
Specially Promoted Research (MEXT, Japan).

Annotations covered by this license must be attributed as follows:

    Corpus annotations (c) GENIA Project

Distributions including annotations covered by this licence must
include this license text and Attribution section.

6. References

- GENIA Project : http://www.geniaproject.org
- PubMed : http://www.pubmed.gov/
- NLM (United States National Library of Medicine) : http://www.nlm.nih.gov/
- MEXT (Ministry of Education, Culture, Sports, Science and Technology) : http://www.mext.go.jp/
- JST (Japan Science and Technology Agency) : http://www.jst.go.jp
"""

GeniaProjectLicense = CustomLicense(
    name="GENIA Project License for Annotated Corpora",
    # NOTE: just in case the link will break
    text=_GENIA_PROJECT_LICENSE_TEXT,
    link="http://www.nactem.ac.uk/meta-knowledge/GENIA_license.txt",
)


_NLM_LICENSE_TEXT = """
National Library of Medicine Terms and Conditions

INTRODUCTION

Downloading data from the National Library of Medicine FTP servers indicates your acceptance of the following Terms and Conditions: No charges, usage fees or royalties are paid to NLM for this data.

GENERAL TERMS AND CONDITIONS

    Users of the data agree to:
        acknowledge NLM as the source of the data by including the phrase "Courtesy of the U.S. National Library of Medicine" in a clear and conspicuous manner,
        properly use registration and/or trademark symbols when referring to NLM products, and
        not indicate or imply that NLM has endorsed its products/services/applications. 

    Users who republish or redistribute the data (services, products or raw data) agree to:
        maintain the most current version of all distributed data, or
        make known in a clear and conspicuous manner that the products/services/applications do not reflect the most current/accurate data available from NLM.

    These data are produced with a reasonable standard of care, but NLM makes no warranties express or implied, including no warranty of merchantability or fitness for particular purpose, regarding the accuracy or completeness of the data. Users agree to hold NLM and the U.S. Government harmless from any liability resulting from errors in the data. NLM disclaims any liability for any consequences due to use, misuse, or interpretation of information contained or not contained in the data.

    NLM does not provide legal advice regarding copyright, fair use, or other aspects of intellectual property rights. See the NLM Copyright page.

    NLM reserves the right to change the type and format of its machine-readable data. NLM will take reasonable steps to inform users of any changes to the format of the data before the data are distributed via the announcement section or subscription to email and RSS updates.

"""
NLMLicense = CustomLicense(
    name="National Library of Medicine Terms and Conditions",
    # NOTE: just in case the link will break
    text=_NLM_LICENSE_TEXT,
    link="https://www.nlm.nih.gov/databases/download/terms_and_conditions.html",
)


_PHYSIONET_LICENSE_1p5_TEXT = """
The PhysioNet Credentialed Health Data License
Version 1.5.0

Copyright (c) 2022 MIT Laboratory for Computational Physiology

The MIT Laboratory for Computational Physiology (MIT-LCP) wishes to make data available for research and educational purposes to qualified requestors, but only if the data are used and protected in accordance with the terms and conditions stated in this License.

It is hereby agreed between the data requestor, hereinafter referred to as the "LICENSEE", and MIT-LCP, that:

    The LICENSEE will not attempt to identify any individual or institution referenced in PhysioNet restricted data.
    The LICENSEE will exercise all reasonable and prudent care to avoid disclosure of the identity of any individual or institution referenced in PhysioNet restricted data in any publication or other communication.
    The LICENSEE will not share access to PhysioNet restricted data with anyone else.
    The LICENSEE will exercise all reasonable and prudent care to maintain the physical and electronic security of PhysioNet restricted data.
    If the LICENSEE finds information within PhysioNet restricted data that he or she believes might permit identification of any individual or institution, the LICENSEE will report the location of this information promptly by email to PHI-report@physionet.org, citing the location of the specific information in question.
    The LICENSEE will use the data for the sole purpose of lawful use in scientific research and no other.
    The LICENSEE will be responsible for ensuring that he or she maintains up to date certification in human research subject protection and HIPAA regulations.
    The LICENSEE agrees to contribute code associated with publications arising from this data to a repository that is open to the research community.
    This agreement may be terminated by either party at any time, but the LICENSEE's obligations with respect to PhysioNet data shall continue after termination.  

THE DATA ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA OR THE USE OR OTHER DEALINGS IN THE DATA.
"""
PhysioNetLicense1p5 = CustomLicense(
    name="PhysioNet Credentialed Health Data License",
    version="1.5",
    link="https://physionet.org/content/mimiciv/view-license/0.4/",
    # NOTE: just in case the link will break
    text=_PHYSIONET_LICENSE_1p5_TEXT,
)


UMLSLicense = CustomLicense(
    name="UMLS - Metathesaurus License Agreement",
    link="https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement.html",
)

# NOTE: just in case link will break
_NCBI_LICENSE_TEXT = """
===========================================================================
*
*                            PUBLIC DOMAIN NOTICE
*               National Center for Biotechnology Information
*
*  This software/database is a "United States Government Work" under the
*  terms of the United States Copyright Act.  It was written as part of
*  the author's official duties as a United States Government employee and
*  thus cannot be copyrighted.  This software/database is freely available
*  to the public for use. The National Library of Medicine and the U.S.
*  Government have not placed any restriction on its use or reproduction.
*
*  Although all reasonable efforts have been taken to ensure the accuracy
*  and reliability of the software and data, the NLM and the U.S.
*  Government do not and cannot warrant the performance or results that
*  may be obtained by using this software or data. The NLM and the U.S.
*  Government disclaim all warranties, express or implied, including
*  warranties of performance, merchantability or fitness for any particular
*  purpose.
*
*  Please cite the author in any work or product based on this material.
*
*
===========================================================================
"""
NCBILicense = CustomLicense(
    name="National Center fr Biotechnology Information PUBLIC DOMAIN NOTICE",
    link="https://github.com/openbiocorpora/genetag/blob/master/LICENSE",
    text=_NCBI_LICENSE_TEXT,
)
