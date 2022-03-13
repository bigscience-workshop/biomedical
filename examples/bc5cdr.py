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
To this end, we set up a challenge task through BioCreative V to automatically
extract CDRs from the literature. More specifically, we designed two challenge
tasks: disease named entity recognition (DNER) and chemical-induced disease
(CID) relation extraction. To assist system development and assessment, we
created a large annotated text corpus that consists of human annotations of
all chemicals, diseases and their interactions in 1,500 PubMed articles.

-- 'Overview of the BioCreative V Chemical Disease Relation (CDR) Task'
"""
import csv
import json
import os

import bioc
import datasets
from dataclasses import dataclass

_CITATION = """\
@article{DBLP:journals/biodb/LiSJSWLDMWL16,
  author    = {Jiao Li and
               Yueping Sun and
               Robin J. Johnson and
               Daniela Sciaky and
               Chih{-}Hsuan Wei and
               Robert Leaman and
               Allan Peter Davis and
               Carolyn J. Mattingly and
               Thomas C. Wiegers and
               Zhiyong Lu},
  title     = {BioCreative {V} {CDR} task corpus: a resource for chemical disease
               relation extraction},
  journal   = {Database J. Biol. Databases Curation},
  volume    = {2016},
  year      = {2016},
  url       = {https://doi.org/10.1093/database/baw068},
  doi       = {10.1093/database/baw068},
  timestamp = {Thu, 13 Aug 2020 12:41:41 +0200},
  biburl    = {https://dblp.org/rec/journals/biodb/LiSJSWLDMWL16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
The BioCreative V Chemical Disease Relation (CDR) dataset is a large annotated text corpus of 
human annotations of all chemicals, diseases and their interactions in 1,500 PubMed articles.
"""

_HOMEPAGE = "http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/"

_LICENSE = "Public Domain Mark 1.0"

_URLs = {
    "source": "http://www.biocreative.org/media/store/files/2016/CDR_Data.zip",
    "bigbio": "http://www.biocreative.org/media/store/files/2016/CDR_Data.zip",
}

_SUPPORTED_TASKS = ["NER", "NED", "RE"]
_SOURCE_VERSION = datasets.Version("01.05.16")
_BIOBIO_VERSION = datasets.Version("1.0.0")


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    schema: str = None  # options = (source|bigbio)
    task_id: str = None  # e.g, bioasq10b


class Bc5cdrDataset(datasets.GeneratorBasedBuilder):
    """
    BioCreative V Chemical Disease Relation (CDR) Task.
    """

    DEFAULT_CONFIG_NAME = "bc5cdr_source"

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bc5cdr_source",
            version=_SOURCE_VERSION,
            description="BC5CDR source schema",
            schema="source",
            task_id="bc5cdr",
        ),
        BigBioConfig(
            name="bc5cdr_bigbio",
            version=_BIOBIO_VERSION,
            description="BC5CDR simplified BigBio schema",
            schema="bigbio_kb",
            task_id="bc5cdr",
        ),
    ]

    def _info(self):

        if self.config.schema == "source":
            # this is a variation on the BioC format
            features = datasets.Features(
                {
                    "passages": [
                        {
                            "document_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "entities": [
                                {
                                    "id": datasets.Value("string"),
                                    "offsets": [[datasets.Value("int32")]],
                                    "text": [datasets.Value("string")],
                                    "type": datasets.Value("string"),
                                    "normalized": [
                                        {
                                            "db_name": datasets.Value("string"),
                                            "db_id": datasets.Value("string"),
                                        }
                                    ],
                                }
                            ],
                            "relations": [
                                {
                                    "id": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                    "arg1_id": datasets.Value("string"),
                                    "arg2_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ]
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
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
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml"
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml"
                    ),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml",
                    ),
                    "split": "dev",
                },
            ),
        ]

    def _get_bioc_entity(self, span, db_id_key="MESH"):
        """
        Load BioC entity
        TODO: Fix the discontiguous entity parsing issue with BioC
        """
        offsets = [(loc.offset, loc.offset + loc.length) for loc in span.locations]
        db_id = span.infons[db_id_key] if db_id_key else -1
        return {
            "id": span.id,
            "offsets": offsets,
            "text": [span.text],
            "type": span.infons["type"],
            "normalized": [{"db_name": db_id_key, "db_id": db_id}],
        }

    def _generate_examples(
        self,
        filepath,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        if self.config.schema == "source":
            reader = bioc.BioCXMLDocumentReader(str(filepath))
            for uid, xdoc in enumerate(reader):
                yield uid,  {
                    "passages": [
                        {   "document_id": xdoc.id,
                            "type": passage.infons["type"],
                            "text": passage.text,
                            "entities": [
                                self._get_bioc_entity(span)
                                for span in passage.annotations
                            ],
                            "relations": [
                                {
                                    "id": rel.id,
                                    "type": rel.infons["relation"],
                                    "arg1_id": rel.infons["Chemical"],
                                    "arg2_id": rel.infons["Disease"],
                                }
                                for rel in xdoc.relations
                            ],
                        }
                        for passage in xdoc.passages
                    ]
                }

        elif self.config.schema == "bigbio_kb":
            reader = bioc.BioCXMLDocumentReader(str(filepath))
            uid = 0  # global unique id

            for i, xdoc in enumerate(reader):
                data = {
                    "id": uid,
                    "document_id": xdoc.id,
                    "passages": [],
                    "entities": [],
                    "relations": [],
                }
                uid += 1

                char_start = 0
                # passages must not overlap and spans must cover the entire document
                for passage in xdoc.passages:
                    offsets = [[char_start, char_start + len(passage.text)]]
                    char_start = char_start + len(passage.text) + 1
                    data["passages"].append(
                        {
                            "id": uid,
                            "type": passage.infons["type"],
                            "text": [passage.text],
                            "offsets": offsets,
                        }
                    )
                    uid += 1

                # entities
                for passage in xdoc.passages:
                    for span in passage.annotations:
                        ent = self._get_bioc_entity(span, db_id_key="MESH")
                        ent["id"] = uid  # override BioC default id
                        data["entities"].append(ent)
                        uid += 1

                # relations
                for rela in xdoc.relations:
                    data["relations"].append(
                        {
                            "id": uid,
                            "type": rela.infons["relation"],
                            "arg1_id": rela.infons["Chemical"],
                            "arg2_id": rela.infons["Disease"],
                            "normalized": [],
                        }
                    )
                    uid += 1

                yield i, data
