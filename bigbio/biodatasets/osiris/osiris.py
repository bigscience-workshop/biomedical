# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
import os
import uuid
import xml.etree.ElementTree as ET
from typing import List

import datasets
from numpy import int32

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = []
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@ARTICLE{Furlong2008,
  author = {Laura I Furlong and Holger Dach and Martin Hofmann-Apitius and Ferran Sanz},
  title = {OSIRISv1.2: a named entity recognition system for sequence variants
  of genes in biomedical literature.},
  journal = {BMC Bioinformatics},
  year = {2008},
  volume = {9},
  pages = {84},
  doi = {10.1186/1471-2105-9-84},
  pii = {1471-2105-9-84},
  pmid = {18251998},
  timestamp = {2013.01.15},
  url = {http://dx.doi.org/10.1186/1471-2105-9-84}
}
"""

_DATASETNAME = "osiris"

_DESCRIPTION = """\
The OSIRIS corpus is a set of MEDLINE abstracts manually annotated
with human variation mentions. The corpus is distributed under the terms
of the Creative Commons Attribution License
Creative Commons Attribution 3.0 Unported License,
which permits unrestricted use, distribution, and reproduction in any medium,
provided the original work is properly cited (Furlong et al, BMC Bioinformatics 2008, 9:84).
"""

_HOMEPAGE = "https://sites.google.com/site/laurafurlongweb/databases-and-tools/corpora/"


_LICENSE = Licenses.CC_BY_3p0

_URLS = {
    _DATASETNAME: [
        "https://github.com/rockt/SETH/blob/master/resources/OSIRIS/corpus.xml?raw=true "
    ]
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]


_SOURCE_VERSION = "1.2.0"

_BIGBIO_VERSION = "1.0.0"


class Osiris(datasets.GeneratorBasedBuilder):
    """
    The OSIRIS corpus is a set of MEDLINE abstracts manually annotated
    with human variation mentions.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="osiris_source",
            version=SOURCE_VERSION,
            description="osiris source schema",
            schema="source",
            subset_id="osiris",
        ),
        BigBioConfig(
            name="osiris_bigbio_kb",
            version=BIGBIO_VERSION,
            description="osiris BigBio schema",
            schema="bigbio_kb",
            subset_id="osiris",
        ),
    ]

    DEFAULT_CONFIG_NAME = "osiris_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "Pmid": datasets.Value("string"),
                    "Title": datasets.Value("string"),
                    "Abstract": datasets.Value("string"),
                    "genes": [
                        {
                            "g_id": datasets.Value("string"),
                            "g_lex": datasets.Value("string"),
                            "offsets": [[datasets.Value("int32")]],
                        }
                    ],
                    "variants": [
                        {
                            "v_id": datasets.Value("string"),
                            "v_lex": datasets.Value("string"),
                            "v_norm": datasets.Value("string"),
                            "offsets": [[datasets.Value("int32")]],
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir[0]),
                    "split": "data",
                },
            )
        ]

    def _get_offsets(self, parent: ET.Element, child: ET.Element) -> List[int32]:
        """
        Retrieves character offsets for child from parent.
        """
        parent_text = " ".join(
            [
                " ".join([t for t in c.itertext()])
                for c in list(parent)
                if c.tag != "Pmid"
            ]
        )
        child_text = " ".join([t for t in child.itertext()])
        start = parent_text.index(child_text)
        end = start + len(child_text)
        return [start, end]

    def _get_dict(self, elem: ET.Element) -> dict:
        """
        Retrieves dict from XML element.
        """
        elem_d = dict()
        for child in elem:
            elem_d[child.tag] = {}
            elem_d[child.tag]["text"] = " ".join([t for t in child.itertext()])

            if child.tag != "Pmid":
                elem_d[child.tag]["offsets"] = self._get_offsets(elem, child)

            for c in child:
                elem_d[c.tag] = []

            for c in child:
                c_dict = c.attrib
                c_dict["offsets"] = self._get_offsets(elem, c)
                elem_d[c.tag].append(c.attrib)

        return elem_d

    def _handle_missing_variants(self, row: dict) -> dict:
        """
        If variant is not present in the row this function adds one variant
        with no data (to make looping though items possible) and returns the new row.
        These mocked variants will be romoved after parsing.
        Otherwise returns unchanged row.
        """

        if row.get("variant", 0) == 0:
            row["variant"] = [
                {"v_id": "", "v_lex": "", "v_norm": "", "offsets": [0, 0]}
            ]
        return row

    def _get_entities(self, row: dict) -> List[dict]:
        """
        Retrieves two lists of dicts for genes and variants.
        After that, chains both together.
        """
        genes = [
            {
                "id": str(uuid.uuid4()),
                "offsets": [gene["offsets"]],
                "text": [gene["g_lex"]],
                "type": "gene",
                "normalized": [{"db_name": "NCBI Gene", "db_id": gene["g_id"]}],
            }
            for gene in row["gene"]
        ]

        variants = [
            {
                "id": str(uuid.uuid4()),
                "offsets": [variant["offsets"]],
                "text": [variant["v_lex"]],
                "type": "variant",
                "normalized": [
                    {
                        "db_name": "HGVS-like" if variant["v_id"] == "No" else "dbSNP",
                        "db_id": variant["v_norm"]
                        if variant["v_id"] == "No"
                        else variant["v_id"],
                    }
                ],
            }
            for variant in row["variant"]
            if variant["v_id"] != ""
        ]
        return list(itertools.chain(genes, variants))

    def _generate_examples(self, filepath, split):

        root = ET.parse(filepath).getroot()
        uid = 0
        if self.config.schema == "source":
            for elem in list(root):
                row = self._get_dict(elem)

                # handling missing variants data
                row = self._handle_missing_variants(row)
                uid += 1
                yield uid, {
                    "Pmid": row["Pmid"]["text"],
                    "Title": {
                        "offsets": [row["Title"]["offsets"]],
                        "text": row["Title"]["text"],
                    },
                    "Abstract": {
                        "offsets": [row["Abstract"]["offsets"]],
                        "text": row["Abstract"]["text"],
                    },
                    "genes": [
                        {
                            "g_id": gene["g_id"],
                            "g_lex": gene["g_lex"],
                            "offsets": [gene["offsets"]],
                        }
                        for gene in row["gene"]
                    ],
                    "variants": [
                        {
                            "v_id": variant["v_id"],
                            "v_lex": variant["v_lex"],
                            "v_norm": variant["v_norm"],
                            "offsets": [variant["offsets"]],
                        }
                        for variant in row["variant"]
                    ],
                }

        elif self.config.schema == "bigbio_kb":

            for elem in list(root):
                row = self._get_dict(elem)

                # handling missing variants data
                row = self._handle_missing_variants(row)
                uid += 1
                yield uid, {
                    "id": str(uid),
                    "document_id": row["Pmid"]["text"],
                    "passages": [
                        {
                            "id": str(uuid.uuid4()),
                            "type": "title",
                            "text": [row["Title"]["text"]],
                            "offsets": [row["Title"]["offsets"]],
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "type": "abstract",
                            "text": [row["Abstract"]["text"]],
                            "offsets": [row["Abstract"]["offsets"]],
                        },
                    ],
                    "entities": self._get_entities(row),
                    "relations": [],
                    "events": [],
                    "coreferences": [],
                }
