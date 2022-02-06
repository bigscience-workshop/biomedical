# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

from collections import defaultdict
import itertools
import os
import re
import tarfile
from typing import Dict, List
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

import datasets
from datasets import Features, Value

_DATASETNAME = "muchmore"

# TODO: home page has a list of publications but its not clear which to choose
# https://muchmore.dfki.de/papers1.htm
# to start, chose the one below.
# Buitelaar, Paul / Declerck, Thierry / Sacaleanu, Bogdan / Vintar, Spela / Raileanu, Diana / Crispi, Claudia: A Multi-Layered, XML-Based Approach to the Integration of Linguistic and Semantic Annotations. In: Proceedings of EACL 2003 Workshop on Language Technology and the Semantic Web (NLPXML’03), Budapest, Hungary, April 2003.

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
_LICENSE = ""

_URLs = {
    "original": [
        "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_english_train_V4.2.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_V4.2.tar.gz",
    ],
    "translation": [
        "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
    ],
    "en_plain": "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
    "de_plain": "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
    "muchmore": [
        "https://muchmore.dfki.de/pubs/springer_english_train_V4.2.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_V4.2.tar.gz",
    ],
}

# took version from annotated file names
_VERSION = "4.2.0"

NATIVE_ENCODING = "ISO-8859-1"

class MuchMoreDataset(datasets.GeneratorBasedBuilder):
    """MuchMore Springer Bilingual Corpus"""

    VERSION = datasets.Version(_VERSION)
    DEFAULT_CONFIG_NAME = _DATASETNAME

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="original",
            version=VERSION,
            description="muchmore: original format",
        ),
        datasets.BuilderConfig(
            name=DEFAULT_CONFIG_NAME,
            version=VERSION,
            description=_DESCRIPTION,
        ),
        datasets.BuilderConfig(
            name="en_plain",
            version=VERSION,
            description="muchmore: plaintext of english abstracts",
        ),
        datasets.BuilderConfig(
            name="de_plain",
            version=VERSION,
            description="muchmore: plaintext of german abstracts",
        ),
        datasets.BuilderConfig(
            name="translation",
            version=VERSION,
            description="muchmore: plaintext of matched english + german abstracts",
        ),
    ]

    # default config produces english annotations at the moment
    def _info(self):

        if self.config.name == "original":
            features = Features({
                "sample_id": Value("string"),
                "corresp": Value("string"),
                "language": Value("string"),
                "abstract": Value("string"),
                "sentences": [{
                    "id": Value("string"),
                    "corresp": Value("string"),
                    "umlsterms": [{
                        "id": Value("string"),
                        "from": Value("string"),
                        "to": Value("string"),
                        "concepts": [{
                            "id": Value("string"),
                            "cui": Value("string"),
                            "preferred": Value("string"),
                            "tui": Value("string"),
                            "mshs": [{
                                "code": Value("string"),
                            }],
                        }],
                    }],
                    "ewnterms": [{
                        "id": Value("string"),
                        "to": Value("string"),
                        "from": Value("string"),
                        "senses": [{
                            "offset": Value("string"),
                        }],
                    }],
                    "semrels": [{
                        "id": Value("string"),
                        "term1": Value("string"),
                        "term2": Value("string"),
                        "reltype": Value("string"),
                    }],
                    "chunks": [{
                        "id": Value("string"),
                        "to": Value("string"),
                        "from": Value("string"),
                        "type": Value("string"),
                    }],
                    "tokens": [{
                        "id": Value("string"),
                        "pos": Value("string"),
                        "lemma": Value("string"),
                        "text": Value("string"),
                    }],
                }]
            })

        elif self.config.name == _DATASETNAME:
            features = Features(
                {
                    "passages": [
                        {
                            "document_id": Value("string"),
                            "type": Value("string"),
                            "text": Value("string"),
                            "snippets": [
                                {
                                    "snippet_id": Value("string"),
                                    "offsets": [[Value("int32")]],
                                    "text": Value("string"),
                                    "type": Value("string"),
                                }
                            ],
                        }
                    ]
                }
            )

        elif self.config.name == "translation":
            features = Features({
                "en": Value("string"),
                "de": Value("string"),
                "metadata": {
                    "sample_id_prefix": Value("string"),
                    "sample_id_en": Value("string"),
                    "sample_id_de": Value("string"),
                }
            })

        elif self.config.name in ["en_plain", "de_plain"]:
            features = Features({
                "sample_id": Value("string"),
                "sample_id_prefix": Value("string"),
                "language": Value("string"),
                "abstract": Value("string"),
            })

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
        data_dirs = dl_manager.download(my_urls)
        # ensure that data_dirs is always a list of string paths
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file_names_and_pointers": itertools.chain(*[
                        dl_manager.iter_archive(data_dir)
                        for data_dir in data_dirs
                    ]),
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

                mshs = [{
                    "code": xmsh.get("code")
                } for xmsh in xconcept.findall("./msh")]

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

            senses = [{
                "offset": xsense.get("offset")
            } for xsense in xewnterm.findall("./sense")]

            ewnterm = {
                "id": xewnterm.get("id"),
                "from": xewnterm.get("from"),
                "to": xewnterm.get("to"),
                "senses": senses,
            }
            ewnterms.append(ewnterm)

        return ewnterms


    @staticmethod
    def _get_semrels_from_xsent(xsent: Element) -> List[Dict[str,str]]:
        xsemrels = xsent.find("./semrels")
        return [{
            "id": xsemrel.get("id"),
            "term1": xsemrel.get("term1"),
            "term2": xsemrel.get("term2"),
            "reltype": xsemrel.get("reltype"),
        } for xsemrel in xsemrels.findall("./semrel")]


    @staticmethod
    def _get_chunks_from_xsent(xsent: Element) -> List[Dict[str,str]]:
        xchunks = xsent.find("./chunks")
        return [{
            "id": xchunk.get("id"),
            "to": xchunk.get("to"),
            "from": xchunk.get("from"),
            "type": xchunk.get("type"),
        } for xchunk in xchunks.findall("./chunk")]


    @staticmethod
    def _get_tokens_from_xsent(xsent: Element) -> List[Dict[str,str]]:
        xtext = xsent.find("./text")
        return [{
            "id": xtoken.get("id"),
            "pos": xtoken.get("pos"),
            "lemma": xtoken.get("lemma"),
            "text": xtoken.text,
        } for xtoken in xtext.findall("./token")]


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
                    print()
                    print(f"skipping {file_name} with no content")
                    print()
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


    def _generate_muchmore_examples(self, file_names_and_pointers):
        """Generate big science canonical dataset.

        """

        def snippets_tokens_from_sents(sentences):
            snippets = []
            for sentence in sentences:
                snippet = [el["text"] for el in sentence["tokens"]]
                snippets.append(snippet)
            return snippets

        def sid_to_text_off(sid, snip_txts_lens):
            ii_sid = int(sid[1:])
            start = sum(snip_txts_lens[:ii_sid-1]) + (ii_sid-1)
            end = start + snip_txts_lens[ii_sid-1]
            return start, end


        for _id, (file_name, fp) in enumerate(file_names_and_pointers):

            content_bytes = fp.read()
            content_str = content_bytes.decode(NATIVE_ENCODING)
            if content_str == "":
                print()
                print(f"skipping {file_name} with no content")
                print()
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
            snip_txts = [' '.join(snip_tok) for snip_tok in snip_toks]
            snip_txts_lens = [len(el) for el in snip_txts]
            text = ' '.join(snip_txts)

            snippets = []
            for ii in range(len(sentences)):
                snippet = {
                    "snippet_id": sentences[ii]["id"],
                    "offsets": [sid_to_text_off(sentences[ii]["id"], snip_txts_lens)],
                    "text": snip_txts[ii],
                    "type": "sentence",
                }
                snippets.append(snippet)

            yield _id, {
                "passages": [
                    {
                        "document_id": xroot.get("id"),
                        "type": "abstract",
                        "text": text,
                        "snippets": snippets,
                    }
                ]
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
            sample_map[sample_id_prefix].append({
                "language": language,
                "sample_id": sample_id,
                "abstract": abstract})

        _id = 0
        for sample_id_prefix, sample_pair in sample_map.items():
            if len(sample_pair) != 2:
                continue
            en_idx = 0 if sample_pair[0]["language"] == "en" else 1
            de_idx = 0 if en_idx == 1 else 1
            yield _id, {
                "en": sample_pair[en_idx]["abstract"],
                "de": sample_pair[de_idx]["abstract"],
                "metadata": {
                    "sample_id_prefix": sample_id_prefix,
                    "sample_id_en": sample_pair[en_idx]["sample_id"],
                    "sample_id_de": sample_pair[de_idx]["sample_id"],
                }
            }
            _id += 1


    def _generate_examples(self, file_names_and_pointers, split):

        if self.config.name == "original":
            genny = self._generate_original_examples(file_names_and_pointers)

        elif self.config.name == "translation":
            genny = self._generate_translation_examples(file_names_and_pointers)

        elif self.config.name == "en_plain":
            for _id, (file_name, fp) in enumerate(file_names_and_pointers):
                yield _id, {
                    "sample_id_prefix": re.sub(".eng.abstr$", "", file_name),
                    "sample_id": file_name,
                    "language": "en",
                    "abstract": fp.read().decode(NATIVE_ENCODING),
                }

        elif self.config.name == "de_plain":
            for _id, (file_name, fp) in enumerate(file_names_and_pointers):
                yield _id, {
                    "sample_id_prefix": re.sub(".ger.abstr$", "", file_name),
                    "sample_id": file_name,
                    "language": "de",
                    "abstract": fp.read().decode(NATIVE_ENCODING),
                }

        elif self.config.name == _DATASETNAME:
            genny = self._generate_muchmore_examples(file_names_and_pointers)


        for _id, sample in genny:
            yield _id, sample
