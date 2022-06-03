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


import os
from typing import List

import datasets
import xml.etree.ElementTree as ET
import uuid
import html

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{Wei2015,
  doi = {10.1155/2015/918710},
  url = {https://doi.org/10.1155/2015/918710},
  year = {2015},
  publisher = {Hindawi Limited},
  volume = {2015},
  pages = {1--7},
  author = {Chih-Hsuan Wei and Hung-Yu Kao and Zhiyong Lu},
  title = {{GNormPlus}: An Integrative Approach for Tagging Genes,  Gene Families,  and Protein Domains},
  journal = {{BioMed} Research International}
}
"""

_DATASETNAME = "citation_gia_test_collection"

_DESCRIPTION = """\
The Citation GIA Test Collection was recently created for gene 
indexing at the NLM and includes 151 PubMed abstracts with both 
mention-level and document-level annotations. 
They are selected because both have a focus on human genes.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: [
        "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/GNormPlus/GNormPlusCorpus.zip"]
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION,
                    Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class CitationGIATestCollection(datasets.GeneratorBasedBuilder):
    """
    The Citation GIA Test Collection was recently created for gene indexing at the NLM and includes 
    151 PubMed abstracts with both mention-level and document-level annotations. 
    They are selected because both have a focus on human genes.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="citation_gia_test_collection_source",
            version=SOURCE_VERSION,
            description="citation_gia_test_collection source schema",
            schema="source",
            subset_id="citation_gia_test_collection",
        ),
        BigBioConfig(
            name="citation_gia_test_collection_bigbio_kb",
            version=BIGBIO_VERSION,
            description="citation_gia_test_collection BigBio schema",
            schema="bigbio_kb",
            subset_id="citation_gia_test_collection",
        )
    ]

    DEFAULT_CONFIG_NAME = "citation_gia_test_collection_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "passages": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ]
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir[0], "GNormPlusCorpus/NLMIAT.BioC.xml"),
                    "split": "NLMIAT",
                },
            ),
        ]

    def _get_entities(self, annot_d: dict) -> dict:
        ''''
        Converts annotation dict to entity dict.
        '''
        ent = {
            "id": str(uuid.uuid4()),
            "type": annot_d["type"],
            "text": [annot_d["text"]],
            "offsets": [annot_d["offsets"]],
            "normalized": [
                {
                    "db_name": "NCBI Gene" if annot_d["type"].isdigit() else "",
                    "db_id": annot_d["type"] if annot_d["type"].isdigit() else "",
                }
            ],
        }

        return ent

    def _get_offsets_entities(child, parent_text: str, child_text: str, offset: int) -> List[int]:
        '''
        Extracts child text offsets from parent text for entities. 
        Some offsets that were present in the datset were wrong mainly because of string encodings.
        Also a little fraction of parent strings doesn't contain its respective child strings. 
        Hence few assertion errors in the entitity offsets checking test. 
        '''
        if child_text in parent_text:
            index = parent_text.index(child_text)
            start = index + offset

        else:
            start = offset
        end = start + len(child_text)

        return [start, end]

    def _process_annot(self, annot: ET.Element, passages: dict) -> dict:
        ''''
        Converts annotation XML Element to Python dict.
        '''
        parent_text = " ".join([p['text'] for p in passages.values()])
        annot_d = dict()
        a_d = {a.tag: a.text for a in annot}

        for a in list(annot):

            if a.tag == "location":
                offset = int(a.attrib["offset"])
                annot_d["offsets"] = self._get_offsets_entities(
                    html.escape(parent_text[offset:]),
                    html.escape(a_d["text"]), offset)

            elif a.tag != "infon":
                annot_d[a.tag] = html.escape(a.text)

            else:
                annot_d[a.attrib["key"]] = html.escape(a.text)
                
        return annot_d

    def _parse_elem(self, elem: ET.Element) -> dict:
        ''''
        Converts document XML Element to Python dict.
        '''
        elem_d = dict()
        passages = dict()
        annotations = elem.findall(".//annotation")
        elem_d["entities"] = []

        for child in elem:
            elem_d[child.tag] = []

        for child in elem:
            if child.tag == "passage":
                elem_d[child.tag].append({c.tag: html.escape(" ".join(list(filter(
                    lambda item: item, [t.strip('\n') for t in c.itertext()])))) for c in child})

            elif child.tag == "id":
                elem_d[child.tag] = html.escape(child.text)

        for passage in elem_d["passage"]:
            infon = passage["infon"]
            passage.pop("infon", None)
            passages[infon] = passage

        elem_d["passages"] = passages
        elem_d.pop('passage', None)

        for a in annotations:
            elem_d["entities"].append(
                self._process_annot(a, elem_d["passages"]))

        return elem_d

    def _generate_examples(self, filepath, split):

        root = ET.parse(filepath).getroot()

        if self.config.schema == "source":
            uid = 0
            for elem in root.findall("document"):
                row = self._parse_elem(elem)
                uid += 1
                passages = row["passages"]
                yield uid,  {
                    "id": str(uid),
                    "passages": [
                        {
                            "id": str(uuid.uuid4()),
                            "type": "title",
                            "text": [passages["title"]["text"]],
                            "offsets": [[
                                int(passages["title"]["offset"]),
                                int(passages["title"]["offset"]) +
                                len(passages["title"]["text"])
                            ]],
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "type": "abstract",
                            "text": [passages["abstract"]["text"]],
                            "offsets": [[
                                int(passages["abstract"]["offset"]),
                                int(passages["abstract"]["offset"]) +
                                len(passages["abstract"]["text"])
                            ]],
                        }
                    ],
                    "entities": [self._get_entities(a) for a in row["entities"]]
                }

        elif self.config.schema == "bigbio_kb":
            uid = 0
            for elem in root.findall("document"):
                row = self._parse_elem(elem)
                uid += 1
                passages = row["passages"]
                yield uid,  {
                    "id": str(uid),
                    "document_id": str(uuid.uuid4()),
                    "passages": [
                        {
                            "id": str(uuid.uuid4()),
                            "type": "title",
                            "text": [passages["title"]["text"]],
                            "offsets": [[
                                int(passages["title"]["offset"]),
                                int(passages["title"]["offset"]) +
                                len(passages["title"]
                                    ["text"])
                            ]],
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "type": "abstract",
                            "text": [passages["abstract"]["text"]],
                            "offsets": [[
                                int(passages["abstract"]["offset"]),
                                int(passages["abstract"]["offset"]) +
                                len(passages["abstract"]["text"])
                            ]],
                        }
                    ],
                    "entities": [self._get_entities(a) for a in row["entities"]],
                    "relations": [],
                    "events": [],
                    "coreferences": []
                }
