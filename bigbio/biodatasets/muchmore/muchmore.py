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
A dataset loader for the MuchMore Springer Bilingual Corpus

homepage

* https://muchmore.dfki.de/resources1.htm

description of annotation format

* https://muchmore.dfki.de/pubs/D4.1.pdf

Four files are distributed

* springer_english_train_plain.tar.gz (english plain text of abstracts)
* springer_german_train_plain.tar.gz (german plain text of abstracts)
* springer_english_train_V4.2.tar.gz (annotated xml in english)
* springer_german_train_V4.2.tar.gz (annotated xml in german)

Each tar file has one member file per abstract.
There are keys to join the english and german files
but there is not a 1-1 mapping between them (i.e. some
english files have no german counterpart and some german
files have no english counterpart). However, there is a 1-1
mapping between plain text and annotations for a given language
(i.e. an abstract in springer_english_train_plain.tar.gz will
also be found in springer_english_train_V4.2.tar.gz)

Counts,

* 15,631 total abstracts
* 7,823 english abstracts
* 7,808 german abstracts
* 6,374 matched (en/de) abstracts
* 1,449 english abstracts with no german
* 1,434 german abstracts with no english

Notes

* Arthroskopie.00130237.eng.abstr.chunkmorph.annotated.xml seems to be empty

"""

import itertools
import os
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List
from xml.etree.ElementTree import Element

import datasets
from datasets import Features, Value

# TODO: home page has a list of publications but its not clear which to choose
# https://muchmore.dfki.de/papers1.htm
# to start, chose the one below.
# Buitelaar, Paul / Declerck, Thierry / Sacaleanu, Bogdan / Vintar, Spela / Raileanu, Diana / Crispi, Claudia: A Multi-Layered, XML-Based Approach to the Integration of Linguistic and Semantic Annotations. In: Proceedings of EACL 2003 Workshop on Language Technology and the Semantic Web (NLPXML’03), Budapest, Hungary, April 2003.
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.POS]
_LANGUAGES = [Lang.EN, Lang.DE]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{,
  author={Paul Buitelaar and Thierry Declerck and Bogdan Sacaleanu and {\vS}pela Vintar and Diana Raileanu and C. Crispi},
  title={A Multi-Layered , XML-Based Approach to the Integration of Linguistic and Semantic Annotations},
  booktitle={Proceedings of EACL 2003 Workshop on Language Technology and the Semantic Web (NLPXML03)},
  year={2003},
  month={April}
}
"""

_DESCRIPTION = """\
The corpus used in the MuchMore project is a parallel corpus of English-German scientific
medical abstracts obtained from the Springer Link web site. The corpus consists
approximately of 1 million tokens for each language. Abstracts are from 41 medical
journals, each of which constitutes a relatively homogeneous medical sub-domain (e.g.
Neurology, Radiology, etc.). The corpus of downloaded HTML documents is normalized in
various ways, in order to produce a clean, plain text version, consisting of a title, abstract
and keywords. Additionally, the corpus was aligned on the sentence level.

Automatic (!) annotation includes: Part-of-Speech; Morphology (inflection and
decomposition); Chunks; Semantic Classes (UMLS: Unified Medical Language System,
MeSH: Medical Subject Headings, EuroWordNet); Semantic Relations from UMLS.
"""

_HOMEPAGE = "https://muchmore.dfki.de/resources1.htm"

# TODO: website says the following, but don't see a specific license
# TODO: add to FAQs about what to do in this situation.

# "The cross-lingual information access prototype system for the medical domain
# will be made publicly accessible through the internet. It provides access to
# multilingual information on the basis of a domain ontology and classification.
# For the main task of multilingual domain modelling, the project will focus
# on German and English. "
_LICENSE = Licenses.UNKNOWN
_URLs = {
    "muchmore_source": [
        "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_english_train_V4.2.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_V4.2.tar.gz",
    ],
    "muchmore_bigbio_kb": [
        "https://muchmore.dfki.de/pubs/springer_english_train_V4.2.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_V4.2.tar.gz",
    ],
    "muchmore_en_bigbio_kb": "https://muchmore.dfki.de/pubs/springer_english_train_V4.2.tar.gz",
    "muchmore_de_bigbio_kb": "https://muchmore.dfki.de/pubs/springer_german_train_V4.2.tar.gz",
    "plain": [
        "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
    ],
    "plain_en": "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
    "plain_de": "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
    "muchmore_bigbio_t2t": [
        "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
    ],
}

# took version from annotated file names
_SOURCE_VERSION = "4.2.0"
_BIGBIO_VERSION = "1.0.0"
_SUPPORTED_TASKS = [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION]

NATIVE_ENCODING = "ISO-8859-1"
FILE_NAME_PATTERN = r"^(.+?)\.(eng|ger)\.abstr(\.chunkmorph\.annotated\.xml)?$"
LANG_MAP = {"eng": "en", "ger": "de"}


class MuchMoreDataset(datasets.GeneratorBasedBuilder):
    """MuchMore Springer Bilingual Corpus"""

    DEFAULT_CONFIG_NAME = "muchmore_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="muchmore_source",
            version=SOURCE_VERSION,
            description="MuchMore source schema",
            schema="source",
            subset_id="muchmore",
        ),
        BigBioConfig(
            name="muchmore_bigbio_kb",
            version=BIGBIO_VERSION,
            description="MuchMore simplified BigBio kb schema",
            schema="bigbio_kb",
            subset_id="muchmore",
        ),
        BigBioConfig(
            name="muchmore_en_bigbio_kb",
            version=BIGBIO_VERSION,
            description="MuchMore simplified BigBio kb schema",
            schema="bigbio_kb",
            subset_id="muchmore_en",
        ),
        BigBioConfig(
            name="muchmore_de_bigbio_kb",
            version=BIGBIO_VERSION,
            description="MuchMore simplified BigBio kb schema",
            schema="bigbio_kb",
            subset_id="muchmore_de",
        ),
        BigBioConfig(
            name="muchmore_bigbio_t2t",
            version=BIGBIO_VERSION,
            description="MuchMore simplified BigBio translation schema",
            schema="bigbio_t2t",
            subset_id="muchmore",
        ),
    ]

    # default config produces english annotations at the moment
    def _info(self):

        if self.config.schema == "source":
            features = Features(
                {
                    "sample_id": Value("string"),
                    "corresp": Value("string"),
                    "language": Value("string"),
                    "abstract": Value("string"),
                    "sentences": [
                        {
                            "id": Value("string"),
                            "corresp": Value("string"),
                            "umlsterms": [
                                {
                                    "id": Value("string"),
                                    "from": Value("string"),
                                    "to": Value("string"),
                                    "concepts": [
                                        {
                                            "id": Value("string"),
                                            "cui": Value("string"),
                                            "preferred": Value("string"),
                                            "tui": Value("string"),
                                            "mshs": [
                                                {
                                                    "code": Value("string"),
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                            "ewnterms": [
                                {
                                    "id": Value("string"),
                                    "to": Value("string"),
                                    "from": Value("string"),
                                    "senses": [
                                        {
                                            "offset": Value("string"),
                                        }
                                    ],
                                }
                            ],
                            "semrels": [
                                {
                                    "id": Value("string"),
                                    "term1": Value("string"),
                                    "term2": Value("string"),
                                    "reltype": Value("string"),
                                }
                            ],
                            "chunks": [
                                {
                                    "id": Value("string"),
                                    "to": Value("string"),
                                    "from": Value("string"),
                                    "type": Value("string"),
                                }
                            ],
                            "tokens": [
                                {
                                    "id": Value("string"),
                                    "pos": Value("string"),
                                    "lemma": Value("string"),
                                    "text": Value("string"),
                                }
                            ],
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        elif self.config.name in ("plain", "plain_en", "plain_de"):
            features = Features(
                {
                    "sample_id": Value("string"),
                    "sample_id_prefix": Value("string"),
                    "language": Value("string"),
                    "abstract": Value("string"),
                }
            )

        elif self.config.schema == "bigbio_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        data_dirs = dl_manager.download(my_urls)
        # ensure that data_dirs is always a list of string paths
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file_names_and_pointers": itertools.chain(
                        *[dl_manager.iter_archive(data_dir) for data_dir in data_dirs]
                    ),
                    "split": "train",
                },
            ),
        ]

    @staticmethod
    def _get_umlsterms_from_xsent(xsent: Element) -> List:
        xumlsterms = xsent.find("./umlsterms")

        umlsterms = []
        for xumlsterm in xumlsterms.findall("./umlsterm"):

            concepts = []
            for xconcept in xumlsterm.findall("./concept"):

                mshs = [
                    {"code": xmsh.get("code")} for xmsh in xconcept.findall("./msh")
                ]

                concept = {
                    "id": xconcept.get("id"),
                    "cui": xconcept.get("cui"),
                    "preferred": xconcept.get("preferred"),
                    "tui": xconcept.get("tui"),
                    "mshs": mshs,
                }
                concepts.append(concept)

            umlsterm = {
                "id": xumlsterm.get("id"),
                "from": xumlsterm.get("from"),
                "to": xumlsterm.get("to"),
                "concepts": concepts,
            }
            umlsterms.append(umlsterm)

        return umlsterms

    @staticmethod
    def _get_ewnterms_from_xsent(xsent: Element) -> List:
        xewnterms = xsent.find("./ewnterms")

        ewnterms = []
        for xewnterm in xewnterms.findall("./ewnterm"):

            senses = [
                {"offset": xsense.get("offset")}
                for xsense in xewnterm.findall("./sense")
            ]

            ewnterm = {
                "id": xewnterm.get("id"),
                "from": xewnterm.get("from"),
                "to": xewnterm.get("to"),
                "senses": senses,
            }
            ewnterms.append(ewnterm)

        return ewnterms

    @staticmethod
    def _get_semrels_from_xsent(xsent: Element) -> List[Dict[str, str]]:
        xsemrels = xsent.find("./semrels")
        return [
            {
                "id": xsemrel.get("id"),
                "term1": xsemrel.get("term1"),
                "term2": xsemrel.get("term2"),
                "reltype": xsemrel.get("reltype"),
            }
            for xsemrel in xsemrels.findall("./semrel")
        ]

    @staticmethod
    def _get_chunks_from_xsent(xsent: Element) -> List[Dict[str, str]]:
        xchunks = xsent.find("./chunks")
        return [
            {
                "id": xchunk.get("id"),
                "to": xchunk.get("to"),
                "from": xchunk.get("from"),
                "type": xchunk.get("type"),
            }
            for xchunk in xchunks.findall("./chunk")
        ]

    @staticmethod
    def _get_tokens_from_xsent(xsent: Element) -> List[Dict[str, str]]:
        xtext = xsent.find("./text")
        return [
            {
                "id": xtoken.get("id"),
                "pos": xtoken.get("pos"),
                "lemma": xtoken.get("lemma"),
                "text": xtoken.text,
            }
            for xtoken in xtext.findall("./token")
        ]

    def _generate_original_examples(self, file_names_and_pointers):
        """Generate something close to the original dataset.

        This will yield one sample per abstract with the plaintext
        and the annotations combined into one object. If an abstract
        is available in both english and german each language version
        will be a distinct example.
        """
        abstracts = {}
        samples = {}
        for file_name, fp in file_names_and_pointers:

            if file_name.endswith(".abstr"):
                sample_id = file_name
                abstracts[sample_id] = fp.read().decode(NATIVE_ENCODING)

            elif file_name.endswith(".abstr.chunkmorph.annotated.xml"):
                content_bytes = fp.read()
                content_str = content_bytes.decode(NATIVE_ENCODING)
                if content_str == "":
                    continue

                xroot = ET.fromstring(content_str)

                sentences = []
                for xsent in xroot.findall("./"):
                    sentence = {
                        "id": xsent.get("id"),
                        "corresp": xsent.get("corresp"),
                        "umlsterms": self._get_umlsterms_from_xsent(xsent),
                        "ewnterms": self._get_ewnterms_from_xsent(xsent),
                        "semrels": self._get_semrels_from_xsent(xsent),
                        "chunks": self._get_chunks_from_xsent(xsent),
                        "tokens": self._get_tokens_from_xsent(xsent),
                    }
                    sentences.append(sentence)

                sample_id = xroot.get("id")
                samples[sample_id] = {
                    "sample_id": sample_id,
                    "corresp": xroot.get("corresp"),
                    "language": xroot.get("lang"),
                    "sentences": sentences,
                }

        for _id, (sample_id, sample) in enumerate(samples.items()):
            sample["abstract"] = abstracts[sample_id]
            yield _id, sample

    def _generate_bigbio_kb_examples(self, file_names_and_pointers):
        """Generate big science biomedical kb examples."""

        def snippets_tokens_from_sents(sentences):
            snippets = []
            for sentence in sentences:
                snippet = [el["text"] for el in sentence["tokens"]]
                snippets.append(snippet)
            return snippets

        def sid_to_text_off(sid, snip_txts_lens):
            ii_sid = int(sid[1:])
            start = sum(snip_txts_lens[: ii_sid - 1]) + (ii_sid - 1)
            end = start + snip_txts_lens[ii_sid - 1]
            return start, end

        def sid_wid_to_text_off(sid, wid, snip_txts_lens, snip_toks_lens):
            s_start, s_end = sid_to_text_off(sid, snip_txts_lens)
            ii_sid = int(sid[1:])
            ii_wid = int(wid[1:])
            w_start = sum(snip_toks_lens[ii_sid - 1][: ii_wid - 1]) + (ii_wid - 1)
            start = s_start + w_start
            end = start + snip_toks_lens[ii_sid - 1][ii_wid - 1]
            return start, end

        for _id, (file_name, fp) in enumerate(file_names_and_pointers):

            content_bytes = fp.read()
            content_str = content_bytes.decode(NATIVE_ENCODING)
            if content_str == "":
                continue

            xroot = ET.fromstring(content_str)

            sentences = []
            for xsent in xroot.findall("./"):
                sentence = {
                    "id": xsent.get("id"),
                    "corresp": xsent.get("corresp"),
                    "umlsterms": self._get_umlsterms_from_xsent(xsent),
                    "ewnterms": self._get_ewnterms_from_xsent(xsent),
                    "semrels": self._get_semrels_from_xsent(xsent),
                    "chunks": self._get_chunks_from_xsent(xsent),
                    "tokens": self._get_tokens_from_xsent(xsent),
                }
                sentences.append(sentence)

            snip_toks = snippets_tokens_from_sents(sentences)
            snip_txts = [" ".join(snip_tok) for snip_tok in snip_toks]
            snip_txts_lens = [len(el) for el in snip_txts]
            snip_toks_lens = [[len(tok) for tok in snip] for snip in snip_toks]
            text = " ".join(snip_txts)
            passages = [
                {
                    "id": "{}-passage-0".format(xroot.get("id")),
                    "type": "abstract",
                    "text": [text],
                    "offsets": [(0, len(text))],
                }
            ]

            entities = []
            for sentence in sentences:
                sid = sentence["id"]
                ii_sid = int(sid[1:])

                for umlsterm in sentence["umlsterms"]:
                    umlsterm_id = umlsterm["id"]
                    entity_id = f"{sid}-{umlsterm_id}"
                    wid_from = umlsterm["from"]
                    wid_to = umlsterm["to"]
                    ii_wid_from = int(wid_from[1:])
                    ii_wid_to = int(wid_to[1:])

                    tok_text = " ".join(
                        snip_toks[ii_sid - 1][ii_wid_from - 1 : ii_wid_to]
                    )
                    w_from_start, w_from_end = sid_wid_to_text_off(
                        sid, wid_from, snip_txts_lens, snip_toks_lens
                    )
                    w_to_start, w_to_end = sid_wid_to_text_off(
                        sid, wid_to, snip_txts_lens, snip_toks_lens
                    )

                    offsets = [(w_from_start, w_to_end)]
                    main_text = text[w_from_start:w_to_end]
                    umls_cuis = [el["cui"] for el in umlsterm["concepts"]]

                    entity = {
                        "id": "{}-{}".format(xroot.get("id"), entity_id),
                        "offsets": offsets,
                        "text": [tok_text],
                        "type": "umlsterm",
                        "normalized": [
                            {"db_name": "UMLS", "db_id": cui} for cui in umls_cuis
                        ],
                    }
                    entities.append(entity)

            yield _id, {
                "id": xroot.get("id"),
                "document_id": xroot.get("id"),
                "passages": passages,
                "entities": entities,
                "coreferences": [],
                "events": [],
                "relations": [],
            }

    def _generate_plain_examples(self, file_names_and_pointers):
        """Generate plain text abstract examples."""
        for _id, (file_name, fp) in enumerate(file_names_and_pointers):
            match = re.match(FILE_NAME_PATTERN, file_name)
            yield _id, {
                "sample_id_prefix": match.group(1),
                "sample_id": file_name,
                "language": LANG_MAP[match.group(2)],
                "abstract": fp.read().decode(NATIVE_ENCODING),
            }

    def _generate_translation_examples(self, file_names_and_pointers):
        sample_map = defaultdict(list)
        for file_name, fp in file_names_and_pointers:
            if file_name.endswith("eng.abstr"):
                language = "en"
            elif file_name.endswith("ger.abstr"):
                language = "de"
            else:
                raise ValueError()
            sample_id_prefix = re.sub(".(eng|ger).abstr$", "", file_name)
            sample_id = file_name
            abstract = fp.read().decode(NATIVE_ENCODING)
            sample_map[sample_id_prefix].append(
                {"language": language, "sample_id": sample_id, "abstract": abstract}
            )

        _id = 0
        for sample_id_prefix, sample_pair in sample_map.items():
            if len(sample_pair) != 2:
                continue
            en_idx = 0 if sample_pair[0]["language"] == "en" else 1
            de_idx = 0 if en_idx == 1 else 1
            yield _id, {
                "id": sample_id_prefix,
                "document_id": sample_id_prefix,
                "text_1": sample_pair[en_idx]["abstract"],
                "text_2": sample_pair[de_idx]["abstract"],
                "text_1_name": "en",
                "text_2_name": "de",
            }
            _id += 1

    def _generate_examples(self, file_names_and_pointers, split):

        if self.config.schema == "source":
            genny = self._generate_original_examples(file_names_and_pointers)

        elif self.config.schema == "bigbio_kb":
            genny = self._generate_bigbio_kb_examples(file_names_and_pointers)

        elif self.config.name in ("plain", "plain_en", "plain_de"):
            genny = self._generate_plain_examples(file_names_and_pointers)

        elif self.config.schema == "bigbio_t2t":
            genny = self._generate_translation_examples(file_names_and_pointers)

        for _id, sample in genny:
            yield _id, sample
