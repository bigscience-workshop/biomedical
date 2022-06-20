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

"""
The data came from the GENIA version 3.02 corpus (Kim et al., 2003).
This was formed from a controlled search on MEDLINE using the MeSH terms human, blood cells and transcription factors.
From this search 2,000 abstracts were selected and hand annotated according to a small taxonomy of 48 classes based on
a chemical classification. Among the classes, 36 terminal classes were used to annotate the GENIA corpus.
"""

from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.GENE, Tags.CELL]
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{collier-kim-2004-introduction,
title = "Introduction to the Bio-entity Recognition Task at {JNLPBA}",
author = "Collier, Nigel and Kim, Jin-Dong",
booktitle = "Proceedings of the International Joint Workshop
on Natural Language Processing in Biomedicine and its Applications
({NLPBA}/{B}io{NLP})",
month = aug # " 28th and 29th", year = "2004",
address = "Geneva, Switzerland",
publisher = "COLING",
url = "https://aclanthology.org/W04-1213",
pages = "73--78",
}
"""

_DATASETNAME = "JNLPBA"

_DESCRIPTION = """\
NER For Bio-Entities
"""

_HOMEPAGE = "http://www.geniaproject.org/shared-tasks/bionlp-jnlpba-shared-task-2004"

_LICENSE = Licenses.CC_BY_3p0

_URLS = {
    _DATASETNAME: "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = "3.2.0"

_BIGBIO_VERSION = "1.0.0"


class JNLPBADataset(datasets.GeneratorBasedBuilder):
    """
    The data came from the GENIA version 3.02 corpus
    (Kim et al., 2003).
    This was formed from a controlled search on MEDLINE
    using the MeSH terms human, blood cells and transcription factors.
    From this search 2,000 abstracts were selected and hand annotated
    according to a small taxonomy of 48 classes based on
    a chemical classification.
    Among the classes, 36 terminal classes were used to annotate the GENIA corpus.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="jnlpba_source",
            version=SOURCE_VERSION,
            description="jnlpba source schema",
            schema="source",
            subset_id="jnlpba",
        ),
        BigBioConfig(
            name="jnlpba_bigbio_kb",
            version=BIGBIO_VERSION,
            description="jnlpba BigBio schema",
            schema="bigbio_kb",
            subset_id="jnlpba",
        ),
    ]

    DEFAULT_CONFIG_NAME = "jnlpba_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.load_dataset("jnlpba", split="train").features

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
        """Returns SplitGenerators."""
        data = datasets.load_dataset("jnlpba")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={"data": data["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data": data["validation"]},
            ),
        ]

    def _generate_examples(self, data: datasets.Dataset) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        uid = 0

        if self.config.schema == "source":
            for key, sample in enumerate(data):
                yield key, sample

        elif self.config.schema == "bigbio_kb":
            for i, sample in enumerate(data):
                feature_dict = {
                    "id": uid,
                    "document_id": "NULL",
                    "passages": [],
                    "entities": [],
                    "relations": [],
                    "events": [],
                    "coreferences": [],
                }

                uid += 1
                offset_start = 0
                for token, tag in zip(sample["tokens"], sample["ner_tags"]):
                    offset_start += len(token) + 1
                    feature_dict["entities"].append(
                        {
                            "id": uid,
                            "offsets": [[offset_start, offset_start + len(token)]],
                            "text": [token],
                            "type": tag,
                            "normalized": [],
                        }
                    )
                    uid += 1

                # entities
                yield i, feature_dict
