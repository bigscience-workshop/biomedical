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
import os
import itertools
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False

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
    "train": "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz",
    "test": "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "3.2.0"

_BIGBIO_VERSION = "1.0.0"

logger = datasets.utils.logging.get_logger(__name__)


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
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-DNA",
                                "I-DNA",
                                "B-RNA",
                                "I-RNA",
                                "B-cell_line",
                                "I-cell_line",
                                "B-cell_type",
                                "I-cell_type",
                                "B-protein",
                                "I-protein",
                            ]
                        )
                    ),
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
        """Returns SplitGenerators."""
        train_filepath = dl_manager.download_and_extract(_URLS["train"])
        test_filepath = dl_manager.download_and_extract(_URLS["test"])
        train_file = os.path.join(train_filepath, "Genia4ERtask1.iob2")
        test_file = os.path.join(test_filepath, "Genia4EReval1.iob2")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": train_file},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_file},
            ),
        ]

    def _parse_sentence(self, tokens, ner_tags, uid):
        """
        This function takes in two stacks, one with tokens and the other with tags
        It returns the passage and the entities as required by the bigbio_kb schema
        """
        entities = []
        sentence_words = []
        distance_from_back = -1
        while tokens:
            curr_token = tokens.pop()
            ner_tag = ner_tags.pop()
            distance_from_back += len(curr_token) + 1
            sentence_words.append(curr_token)
            if ner_tag.startswith("I-"):
                tag_text = [curr_token]
                tag_end = distance_from_back - (len(curr_token) + 1)
                curr_tag = ner_tag[2:]
                while not ner_tag.startswith("B-"):
                    curr_token = tokens.pop()
                    ner_tag = ner_tags.pop()
                    distance_from_back += len(curr_token) + 1
                    sentence_words.append(curr_token)
                    tag_text.append(curr_token)
                tag_text = " ".join(list(reversed(tag_text)))
                tag_start = tag_end - len(tag_text)
                entity = {
                    "id": next(uid),
                    "type": curr_tag,
                    "text": [tag_text],
                    "normalized": [],
                    "offsets": [[tag_start, tag_end]],
                }
                entities.append(entity)
            elif ner_tag.startswith("B-"):
                tag_end = distance_from_back
                curr_tag = ner_tag[2:]
                tag_start = tag_end - len(curr_token)
                entity = {
                    "id": next(uid),
                    "type": curr_tag,
                    "text": [curr_token],
                    "normalized": [],
                    "offsets": [[tag_start, tag_end]],
                }
                entities.append(entity)
            elif ner_tag == "O":
                continue
        passage = " ".join(list(reversed(sentence_words)))
        for entity in entities:
            entity_start = len(passage) - entity["offsets"][0][1]
            entity_end = len(passage) - entity["offsets"][0][0]
            entity["offsets"][0][1] = entity_end
            entity["offsets"][0][0] = entity_start

        document = {}
        document["id"] = next(uid)
        document["document_id"] = document["id"]
        document["entities"] = entities
        document["passages"] = [
            {
                "id": next(uid),
                "type": None,
                "text": [passage],
                "offsets": [[0, len(passage)]],
            }
        ]
        document["relations"] = []
        document["events"] = []
        document["coreferences"] = []
        return document["id"], document

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        logger.info(f"Generating examples from {filepath}")
        uid = itertools.count(0)
        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as f:
                tokens = []
                ner_tags = []
                for line in f:
                    if line == "" or line == "\n":
                        if tokens:
                            id = next(uid)
                            yield id, {
                                "id": id,
                                "tokens": tokens,
                                "ner_tags": ner_tags,
                            }
                            next(uid)
                            tokens = []
                            ner_tags = []

                    else:
                        # tokens are tab separated
                        splits = line.split("\t")
                        tokens.append(splits[0])
                        ner_tags.append(splits[1].rstrip())
                # last example
                id = next(uid)
                yield id, {
                    "id": id,
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }

        elif self.config.schema == "bigbio_kb":
            with open(filepath, encoding="utf-8") as f:
                tokens = []
                ner_tags = []
                for line in f:
                    if line == "" or line == "\n":
                        document_id, document = self._parse_sentence(
                            tokens, ner_tags, uid
                        )
                        yield document_id, document
                    else:
                        token, tag = line.split("\t")
                        tokens.append(token.strip())
                        ner_tags.append(tag.strip())
