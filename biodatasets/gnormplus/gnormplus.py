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

import itertools
import os
from typing import Dict, List, Tuple

import datasets
from bioc import biocxml

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks
from utils.parsing import get_texts_and_offsets_from_bioc_ann

_CITATION = """\
@Article{Wei2015,
author={Wei, Chih-Hsuan and Kao, Hung-Yu and Lu, Zhiyong},
title={GNormPlus: An Integrative Approach for Tagging Genes, Gene Families, and Protein Domains},
journal={BioMed Research International},
year={2015},
month={Aug},
day={25},
publisher={Hindawi Publishing Corporation},
volume={2015},
pages={918710},
issn={2314-6133},
doi={10.1155/2015/918710},
url={https://doi.org/10.1155/2015/918710}
}
"""

_DATASETNAME = "gnormplus"

_DESCRIPTION = """\
We re-annotated two existing gene corpora. The BioCreative II GN corpus is a widely used data set for benchmarking GN
tools and includes document-level annotations for a total of 543 articles (281 in its training set; and 262 in test).
The Citation GIA Test Collection was recently created for gene indexing at the NLM and includes 151 PubMed abstracts
with both mention-level and document-level annotations. They are selected because both have a focus on human genes.
For both corpora, we added annotations of gene families and protein domains. For the BioCreative GN corpus, we also
added mention-level gene annotations. As a result, in our new corpus, there are a total of 694 PubMed articles.
PubTator was used as our annotation tool along with BioC formats.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/"

_LICENSE = ""

_URLS = {
    _DATASETNAME: "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/GNormPlus/GNormPlusCorpus.zip"
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class GnormplusDataset(datasets.GeneratorBasedBuilder):
    """Dataset loader for GNormPlus corpus."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="gnormplus_source",
            version=SOURCE_VERSION,
            description="gnormplus source schema",
            schema="source",
            subset_id="gnormplus",
        ),
        BigBioConfig(
            name="gnormplus_bigbio_kb",
            version=BIGBIO_VERSION,
            description="gnormplus BigBio schema",
            schema="bigbio_kb",
            subset_id="gnormplus",
        ),
    ]

    DEFAULT_CONFIG_NAME = "gnormplus_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "passages": [
                        {
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "location": {"offset": datasets.Value("int64"), "length": datasets.Value("int64")},
                        }
                    ],
                    "entities": [
                        {
                            "locations": [{"offset": datasets.Value("int64"), "length": datasets.Value("int64")}],
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "normalized": [{"db_id": datasets.Value("string")}],
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
        else:
            raise NotImplementedError(self.config.schema)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "GNormPlusCorpus/BC2GNtrain.BioC.xml"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "GNormPlusCorpus/BC2GNtest.BioC.xml"),
                },
            ),
        ]

    def _parse_bioc_entity(self, uid, bioc_ann, db_id_key="NCBI"):
        offsets, texts = get_texts_and_offsets_from_bioc_ann(bioc_ann)
        normalized = []
        _type = bioc_ann.infons["type"]
        if _type in bioc_ann.infons:
            db_ids = bioc_ann.infons[_type]
            normalized = [
                # some entities are linked to multiple normalized ids
                {"db_name": db_id_key, "db_id": _id}
                for _id in db_ids.split(",")
            ]
        return {
            "id": uid,
            "offsets": offsets,
            "text": texts,
            "type": _type,
            "normalized": normalized,
        }

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        uid = map(str, itertools.count(start=0, step=1))

        with open(filepath, "r") as fp:
            collection = biocxml.load(fp)

            for idx, document in enumerate(collection.documents):
                if self.config.schema == "source":
                    features = {
                        "doc_id": document.id,
                        "passages": [
                            {
                                "text": passage.text,
                                "type": passage.infons["type"],
                                "location": {"offset": passage.offset, "length": passage.total_span.length},
                            }
                            for passage in document.passages
                        ],
                        "entities": [
                            {
                                "locations": [
                                    {"offset": loc.offset, "length": loc.length} for loc in entity.locations
                                ],
                                "text": entity.text,
                                "type": entity.infons["type"],
                                "normalized": [
                                    {
                                        "db_id": entity.infons[entity.infons["type"]],
                                    }
                                ]
                                if entity.infons["type"] in entity.infons
                                else [],
                            }
                            for passage in document.passages
                            for entity in passage.annotations
                        ],
                    }
                    yield idx, features
                elif self.config.schema == "bigbio_kb":
                    # passage offsets/lengths do not connect, recalculate them for this schema.
                    passage_spans = []
                    start = 0
                    for passage in document.passages:
                        end = start + len(passage.text)
                        passage_spans.append((start, end))
                        start = end + 1

                    features = {
                        "id": next(uid),
                        "document_id": document.id,
                        "passages": [
                            {
                                "id": next(uid),
                                "type": passage.infons["type"],
                                "text": [passage.text],
                                "offsets": [span],
                            }
                            for passage, span in zip(document.passages, passage_spans)
                        ],
                        "entities": [
                            self._parse_bioc_entity(next(uid), entity)
                            for passage in document.passages
                            for entity in passage.annotations
                        ],
                        "events": [],
                        "coreferences": [],
                        "relations": [],
                    }
                    yield idx, features
                else:
                    raise NotImplementedError(self.config.schema)
