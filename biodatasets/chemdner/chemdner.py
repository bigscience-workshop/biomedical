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
import re
from typing import Dict, Iterator, List, Tuple

import bioc
import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@Article{Krallinger2015,
author={Krallinger,
Martin and Rabal,
Obdulia and Leitner,
Florian and Vazquez,
Miguel and Salgado,
David and Lu,
Zhiyong and Leaman,
Robert and Lu,
Yanan and Ji,
Donghong and Lowe,
Daniel M. and Sayle,
Roger A. and Batista-Navarro,
Riza Theresa and Rak,
Rafal and Huber,
Torsten and Rockt{\"a}schel,
Tim and Matos,
S{\'e}rgio and Campos,
David and Tang,
Buzhou and Xu,
Hua and Munkhdalai,
Tsendsuren and Ryu,
Keun Ho and Ramanan,
S. V. and Nathan,
Senthil and {\v{Z}}itnik,
Slavko and Bajec,
Marko and Weber,
Lutz and Irmer,
Matthias and Akhondi,
Saber A. and Kors,
Jan A. and Xu,
Shuo and An,
Xin and Sikdar,
Utpal Kumar and Ekbal,
Asif and Yoshioka,
Masaharu and Dieb,
Thaer M. and Choi,
Miji and Verspoor,
Karin and Khabsa,
Madian and Giles,
C. Lee and Liu,
Hongfang and Ravikumar,
Komandur Elayavilli and Lamurias,
Andre and Couto,
Francisco M. and Dai,
Hong-Jie and Tsai,
Richard Tzong-Han and Ata,
Caglar and Can,
Tolga and Usi{\'e},
Anabel and Alves,
Rui and Segura-Bedmar,
Isabel and Mart{\'i}nez,
Paloma and Oyarzabal,
Julen and Valencia,
Alfonso},
title={The CHEMDNER corpus of chemicals and drugs and its annotation principles},
journal={Journal of Cheminformatics},
year={2015},
month={Jan},
day={19},
volume={7},
number={1},
pages={S2},
abstract={
    The automatic extraction of chemical information from text requires the recognition of chemical entity mentions
    as one of its key steps. When developing supervised named entity recognition (NER) systems, the availability of
    a large, manually annotated text corpus is desirable. Furthermore, large corpora permit the robust evaluation
    and comparison of different approaches that detect chemicals in documents. We present the CHEMDNER corpus,
    a collection of 10,000 PubMed abstracts that contain a total of 84,355 chemical entity mentions labeled manually
    by expert chemistry literature curators, following annotation guidelines specifically defined for this task.
    The abstracts of the CHEMDNER corpus were selected to be representative for all major chemical disciplines.
    Each of the chemical entity mentions was manually labeled according to its structure-associated chemical entity
    mention (SACEM) class: abbreviation, family, formula, identifier, multiple, systematic and trivial.
    The difficulty and consistency of tagging chemicals in text was measured using an agreement study between
    annotators, obtaining a percentage agreement of 91. For a subset of the CHEMDNER corpus
    (the test set of 3,000 abstracts) we provide not only the Gold Standard manual annotations, but also
    mentions automatically detected by the 26 teams that participated in the BioCreative IV CHEMDNER
    chemical mention recognition task. In addition, we release the CHEMDNER silver standard corpus of automatically
    extracted mentions from 17,000 randomly selected PubMed abstracts. A version of the CHEMDNER corpus in
    the BioC format has been generated as well. We propose a standard for required minimum information about
    entity annotations for the construction of domain specific corpora on chemical and drug entities.
    The CHEMDNER corpus and annotation guidelines are available at:
    http://www.biocreative.org/resources/biocreative-iv/chemdner-corpus/
},
issn={1758-2946},
doi={10.1186/1758-2946-7-S1-S2},
url={https://doi.org/10.1186/1758-2946-7-S1-S2}
}
"""

_DESCRIPTION = """\
We present the CHEMDNER corpus, a collection of 10,000 PubMed abstracts that contain a total of 84,355 chemical entity
mentions labeled manually by expert chemistry literature curators, following annotation guidelines specifically
defined for this task. The abstracts of the CHEMDNER corpus were selected to be representative for all major chemical
disciplines. Each of the chemical entity mentions was manually labeled according to its structure-associated chemical
entity mention (SACEM) class: abbreviation, family, formula, identifier, multiple, systematic and trivial.
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/resources/biocreative-iv/chemdner-corpus/"
_LICENSE = "Creative Commons Attribution 4.0 International"

_URLs = {
    "source": "https://biocreative.bioinformatics.udel.edu/media/store/files/2014/chemdner_corpus.tar.gz",
    "bigbio_kb": "https://biocreative.bioinformatics.udel.edu/media/store/files/2014/chemdner_corpus.tar.gz",
}
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class CHEMDNERDataset(datasets.GeneratorBasedBuilder):
    """CHEMDNER"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="chemdner_source",
            version=SOURCE_VERSION,
            description="CHEMDNER source schema",
            schema="source",
            subset_id="chemdner",
        ),
        BigBioConfig(
            name="chemdner_bigbio_kb",
            version=BIGBIO_VERSION,
            description="CHEMDNER BigBio schema",
            schema="bigbio_kb",
            subset_id="chemdner",
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        "chemdner_source"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

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
                            "offset": datasets.Value("int32"),
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
        data_dir = dl_manager.download_and_extract(my_urls) + "/chemdner_corpus/"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "training.bioc.xml"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "evaluation.bioc.xml"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "development.bioc.xml"),
                    "split": "dev",
                },
            ),
        ]

    def _get_passages_and_entities(self, d: bioc.BioCDocument) -> Tuple[List[Dict], List[List[Dict]]]:

        passages: List[Dict] = []
        entities: List[List[Dict]] = []

        text_total_length = 0

        po_start = 0

        for _, p in enumerate(d.passages):

            # annotation used only for document indexing
            if p.text is None or len(p.text) <= 0:
                continue

            # annotation used only for document indexing
            if p.annotations is None or len([a for a in p.annotations if a.text is not None and len(a.text) > 0]) <= 0:
                continue

            eo = p.offset - text_total_length

            text_total_length += len(p.text) + 1

            po_end = po_start + len(p.text)

            dp = {
                "text": p.text,
                "type": p.infons.get("type"),
                "offsets": [(po_start, po_end)],
                "offset": p.offset,  # original offset
            }

            po_start = po_end + 1

            passages.append(dp)

            pe = []

            for a in p.annotations:

                if a.text is None or len(a.text) <= 0:
                    continue

                a_type = a.infons.get("class")
                # a_type = a.infons.get("type")

                # no in-text annotation: only for document indexing
                if a_type in ["MeSH_Indexing_Chemical", "OTHER"]:
                    continue

                da = {
                    "type": a_type,
                    "offsets": [
                        (
                            loc.offset - eo,
                            loc.offset + loc.length - eo,
                        )
                        for loc in a.locations
                    ],
                    "text": (a.text,),
                    "id": a.id,
                    "normalized": self._get_normalized(a),
                }

                pe.append(da)

            entities.append(pe)

        return passages, entities

    def _get_normalized(self, a: bioc.BioCAnnotation) -> List[Dict]:
        """
        Get normalization DB and ID from annotation identifiers
        """

        identifiers = a.infons.get("identifier")

        if identifiers is not None:

            identifiers = re.split(r",|;", identifiers)

            identifiers = [i for i in identifiers if i != "-"]

            normalized = [i.split(":") for i in identifiers]

            normalized = [{"db_name": elems[0], "db_id": elems[1]} for elems in normalized]

        else:

            normalized = []

        return normalized

    def _generate_examples(
        self,
        filepath: str,
        split: str,
    ) -> Iterator[Tuple[int, Dict]]:
        """Yields examples as (key, example) tuples."""

        reader = bioc.BioCXMLDocumentReader(str(filepath))

        if self.config.schema == "source":

            for uid, doc in enumerate(reader):

                passages, passages_entities = self._get_passages_and_entities(doc)

                if len(passages) < 1 or len(passages_entities) < 1 or len(passages_entities) != len(passages):
                    continue

                for p, pe in zip(passages, passages_entities):

                    p.pop("offsets")  # BioC has only start for passages offsets

                    p["document_id"] = doc.id
                    p["entities"] = pe  # BioC has per passage entities

                yield uid, {"passages": passages}

        elif self.config.schema == "bigbio_kb":
            uid = 0
            for idx, doc in enumerate(reader):

                passages, passages_entities = self._get_passages_and_entities(doc)

                if len(passages) < 1 or len(passages_entities) < 1 or len(passages_entities) != len(passages):
                    continue

                # global id
                uid += 1

                # unpack per-passage entities
                entities = [e for pe in passages_entities for e in pe]

                for p in passages:
                    p.pop("offset")  # drop original offset
                    p["text"] = (p["text"],)  # text in passage is Sequence
                    p["id"] = uid  # override BioC default id
                    uid += 1

                for e in entities:
                    e["id"] = uid  # override BioC default id
                    uid += 1

                yield idx, {
                    "id": uid,
                    "document_id": doc.id,
                    "passages": passages,
                    "entities": entities,
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }
