# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
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
from typing import Dict, List, Tuple

import datasets
from bioc import biocxml

from .bigbiohub import BigBioConfig, Tasks, kb_features

_LOCAL = True
_CITATION = """\
@article{10.1093/jamiaopen/ooab025,
    author = {Kittner, Madeleine and Lamping, Mario and Rieke, Damian T and Götze, Julian and Bajwa, Bariya and
    Jelas, Ivan and Rüter, Gina and Hautow, Hanjo and Sänger, Mario and Habibi, Maryam and Zettwitz, Marit and
    Bortoli, Till de and Ostermann, Leonie and Ševa, Jurica and Starlinger, Johannes and Kohlbacher, Oliver and
    Malek, Nisar P and Keilholz, Ulrich and Leser, Ulf},
    title = "{Annotation and initial evaluation of a large annotated German oncological corpus}",
    journal = {JAMIA Open},
    volume = {4},
    number = {2},
    year = {2021},
    month = {04},
    issn = {2574-2531},
    doi = {10.1093/jamiaopen/ooab025},
    url = {https://doi.org/10.1093/jamiaopen/ooab025},
    note = {ooab025},
    eprint = {https://academic.oup.com/jamiaopen/article-pdf/4/2/ooab025/38830128/ooab025.pdf},
}
"""
_DESCRIPTION = """\
BRONCO150 is a corpus containing selected sentences of 150 German discharge summaries of cancer patients (hepatocelluar
carcinoma or melanoma) treated at Charite Universitaetsmedizin Berlin or Universitaetsklinikum Tuebingen. All discharge
summaries were manually anonymized. The original documents were scrambled at the sentence level to make reconstruction
of individual reports impossible.
"""
_HOMEPAGE = "https://www2.informatik.hu-berlin.de/~leser/bronco/index.html"
_LICENSE = "DUA"
_URLS = {}
_PUBMED = False
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"
_DATASETNAME = "bronco"
_DISPLAYNAME = "BRONCO"
_LANGUAGES = ["German"]


class Bronco(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)
    DEFAULT_CONFIG_NAME = "bronco_bigbio_kb"
    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bronco_source",
            version=SOURCE_VERSION,
            description="BRONCO source schema",
            schema="source",
            subset_id="bronco",
        ),
        BigBioConfig(
            name="bronco_bigbio_kb",
            version=BIGBIO_VERSION,
            description="BRONCO BigBio schema",
            schema="bigbio_kb",
            subset_id="bronco",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "passage": {
                        "offset": datasets.Value("int32"),
                        "text": datasets.Value("string"),
                        "annotation": [
                            {
                                "id": datasets.Value("string"),
                                "infon": {
                                    "file": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                },
                                "location": [
                                    {
                                        "offset": datasets.Value("int32"),
                                        "length": datasets.Value("int32"),
                                    }
                                ],
                                "text": datasets.Value("string"),
                            }
                        ],
                        "relation": [
                            {
                                "id": datasets.Value("string"),
                                "infon": {
                                    "file": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                    "norm/atr": datasets.Value("string"),
                                    "string": datasets.Value("string"),
                                },
                                "node": [
                                    {
                                        "refid": datasets.Value("string"),
                                        "role": datasets.Value("string"),
                                    }
                                ],
                            }
                        ],
                    },
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "bioCFiles", "BRONCO150.xml"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r") as fp:
            data = biocxml.load(fp).documents

            if self.config.schema == "source":
                for uid, doc in enumerate(data):
                    out = {
                        "id": doc.id,
                        "passage": {
                            "offset": doc.passages[0].offset,
                            "text": doc.passages[0].text,
                            "annotation": [],
                            "relation": [],
                        },
                    }

                    # handle entities
                    for annotation in doc.passages[0].annotations:
                        anno = {
                            "id": annotation.id,
                            "infon": annotation.infons,
                            "text": annotation.text,
                            "location": [],
                        }
                        for location in annotation.locations:
                            anno["location"].append(
                                {
                                    "offset": location.offset,
                                    "length": location.length,
                                }
                            )
                        out["passage"]["annotation"].append(anno)

                    # handle relations
                    for relation in doc.passages[0].relations:
                        rel = {
                            "id": relation.id,
                            "node": [],
                        }

                        # relation.infons has different keys depending on the relation type
                        # these must be unified to comply with a fixed schema
                        if relation.infons["type"] == "Normalization":
                            rel["infon"] = {
                                "file": relation.infons["file"],
                                "type": relation.infons["type"],
                                "norm/atr": relation.infons["normalization type"],
                                "string": relation.infons["string"],
                            }
                        else:
                            rel["infon"] = {
                                "file": relation.infons["file"],
                                "type": relation.infons["type"],
                                "norm/atr": relation.infons["attribute type"],
                                "string": "",
                            }

                        for node in relation.nodes:
                            rel["node"].append(
                                {
                                    "refid": node.refid,
                                    "role": node.role,
                                }
                            )

                        out["passage"]["relation"].append(rel)

                    yield uid, out

            elif self.config.schema == "bigbio_kb":
                # reorder the documents so they appear in increasing order
                ordered_data = [data[2], data[4], data[0], data[3], data[1]]
                for uid, doc in enumerate(ordered_data):
                    out = {
                        "id": uid,
                        "document_id": doc.id,
                        "passages": [],
                        "entities": [],
                        "events": [],
                        "coreferences": [],
                        "relations": [],
                    }

                    # catch all normalized entities for lookup
                    norm_map = {}
                    for rel in doc.passages[0].relations:
                        if rel.infons["type"] == "Normalization":
                            norm_map[rel.nodes[0].role] = rel.nodes[0].refid

                    # handle passages - split text into sentences
                    for i, passage in enumerate(doc.passages[0].text.split("\n")):
                        # match the offsets on the text after removing \n
                        if i == 0:
                            marker = 0
                        else:
                            marker = out["passages"][-1]["offsets"][-1][-1] + 1

                        out["passages"].append(
                            {
                                "id": f"{uid}-{i}",
                                "text": [passage],
                                "type": "sentence",
                                "offsets": [[marker, marker + len(passage)]],
                            }
                        )

                    # handle entities
                    for ent in doc.passages[0].annotations:
                        offsets = []
                        text_s = []
                        for loc in ent.locations:
                            offsets.append([loc.offset, loc.offset + loc.length])
                            text_s.append(doc.passages[0].text[loc.offset: loc.offset + loc.length])

                        out["entities"].append(
                            {
                                "id": f"{uid}-{ent.id}",
                                "type": ent.infons["type"],
                                "text": text_s,
                                "offsets": offsets,
                                "normalized": [
                                    {
                                        "db_name": norm_map.get(ent.id, ":").split(":")[0],
                                        # replace faulty connectors in db_ids
                                        "db_id": norm_map.get(ent.id, ":")
                                        .split(":")[1]
                                        .replace(",", ".")
                                        .replace("+", ""),
                                    }
                                ],
                            }
                        )

                    yield uid, out