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
import collections
import itertools
import os

import datasets
from bioc import biocxml

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses
from bigbio.utils.parsing import get_texts_and_offsets_from_bioc_ann

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
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

_DATASETNAME = "bc5cdr"

_DESCRIPTION = """\
The BioCreative V Chemical Disease Relation (CDR) dataset is a large annotated text corpus of
human annotations of all chemicals, diseases and their interactions in 1,500 PubMed articles.
"""

_HOMEPAGE = "http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/"

_LICENSE = Licenses.PUBLIC_DOMAIN_MARK_1p0

_URLs = {
    "source": "http://www.biocreative.org/media/store/files/2016/CDR_Data.zip",
    "bigbio_kb": "http://www.biocreative.org/media/store/files/2016/CDR_Data.zip",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
    Tasks.RELATION_EXTRACTION,
]
_SOURCE_VERSION = "01.05.16"
_BIGBIO_VERSION = "1.0.0"


class Bc5cdrDataset(datasets.GeneratorBasedBuilder):
    """
    BioCreative V Chemical Disease Relation (CDR) Task.
    """

    DEFAULT_CONFIG_NAME = "bc5cdr_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bc5cdr_source",
            version=SOURCE_VERSION,
            description="BC5CDR source schema",
            schema="source",
            subset_id="bc5cdr",
        ),
        BigBioConfig(
            name="bc5cdr_bigbio_kb",
            version=BIGBIO_VERSION,
            description="BC5CDR simplified BigBio schema",
            schema="bigbio_kb",
            subset_id="bc5cdr",
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
            features = schemas.kb_features

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
        my_urls = _URLs[self.config.schema]
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

    def _get_bioc_entity(self, span, doc_text, db_id_key="MESH"):
        """Parse BioC entity annotation.

        Parameters
        ----------
        span : BioCAnnotation
            BioC entity annotation
        doc_text : string
            document text, required to construct text spans
        db_id_key : str, optional
            database name used for normalization, by default "MESH"

        Returns
        -------
        dict
            entity information
        """
        # offsets = [(loc.offset, loc.offset + loc.length) for loc in span.locations]
        # texts = [doc_text[i:j] for i, j in offsets]
        offsets, texts = get_texts_and_offsets_from_bioc_ann(span)
        db_ids = span.infons[db_id_key] if db_id_key else "-1"

        # some entities are not linked and
        # some entities are linked to multiple normalized ids
        if db_ids == "-1":
            db_ids_list = []
        else:
            db_ids_list = db_ids.split("|")

        normalized = [{"db_name": db_id_key, "db_id": db_id} for db_id in db_ids_list]

        return {
            "id": span.id,
            "offsets": offsets,
            "text": texts,
            "type": span.infons["type"],
            "normalized": normalized,
        }

    def _get_relations(self, relations, entities):
        """
        BC5CDR provides abstract-level annotations for entity-linked relation
        pairs rather than materializing links between all surface form
        mentions of relations. An example from train id=2670794, the relation
            - (chemical, disease) (D014148, D004211)
        is materialized as 6 mentions of entity pairs
            - 2x ('tranexamic acid', 'intravascular coagulation')
            - 4x ('AMCA', 'intravascular coagulation')
        """
        # index entities by normalized id
        index = collections.defaultdict(list)
        for ent in entities:
            for norm in ent["normalized"]:
                index[norm["db_id"]].append(ent)
        index = dict(index)

        # transform doc-level relations to mention-level
        rela_mentions = []
        for rela in relations:
            arg1 = rela.infons["Chemical"]
            arg2 = rela.infons["Disease"]
            # all mention pairs
            all_pairs = itertools.product(index[arg1], index[arg2])
            for a, b in all_pairs:
                # create relations linked by entity ids
                rela_mentions.append(
                    {
                        "id": None,
                        "type": rela.infons["relation"],
                        "arg1_id": a["id"],
                        "arg2_id": b["id"],
                        "normalized": [],
                    }
                )
        return rela_mentions

    def _get_document_text(self, xdoc):
        """Build document text for unit testing entity span offsets."""
        text = ""
        for passage in xdoc.passages:
            pad = passage.offset - len(text)
            text += (" " * pad) + passage.text
        return text

    def _generate_examples(
        self,
        filepath,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            reader = biocxml.BioCXMLDocumentReader(str(filepath))

            for uid, xdoc in enumerate(reader):
                doc_text = self._get_document_text(xdoc)
                yield uid, {
                    "passages": [
                        {
                            "document_id": xdoc.id,
                            "type": passage.infons["type"],
                            "text": passage.text,
                            "entities": [
                                self._get_bioc_entity(span, doc_text)
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
            reader = biocxml.BioCXMLDocumentReader(str(filepath))
            uid = 0  # global unique id

            for i, xdoc in enumerate(reader):
                data = {
                    "id": uid,
                    "document_id": xdoc.id,
                    "passages": [],
                    "entities": [],
                    "relations": [],
                    "events": [],
                    "coreferences": [],
                }
                uid += 1
                doc_text = self._get_document_text(xdoc)

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
                        ent = self._get_bioc_entity(span, doc_text, db_id_key="MESH")
                        ent["id"] = uid  # override BioC default id
                        data["entities"].append(ent)
                        uid += 1

                # relations
                relations = self._get_relations(xdoc.relations, data["entities"])
                for rela in relations:
                    rela["id"] = uid  # assign unique id
                    data["relations"].append(rela)
                    uid += 1

                yield i, data
