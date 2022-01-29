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
This is a template on how to implement a dataset in the biomedical repo.

A thorough walkthrough on how to implement a dataset can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

This script corresponds to Step 4 in the Biomedical Hackathon guide.

To start, copy this template file and save it as <your_dataset_name>.py in an appropriately named folder within datasets. Then, modify this file as necessary to implement your own method of extracting, and generating examples for your dataset.

There are 3 key elements to implementing a dataset:

(1) `_info`: Create a skeletal structure that describes what is in the dataset and the nature of the features.

(2) `_split_generators`: Download and extract data for each split of the data (ex: train/dev/test)

(3) `_generate_examples`: From downloaded + extracted data, process files for the data in a feature format specified in "info".

----------------------
Step 1: Declare imports
Your imports go here; the only mandatory one is `datasets`, as methods and attributes from this library will be used throughout the script.

We have provided some import statements that we strongly recommend. Feel free to adapt; so long as the style-guide requirements are satisfied (Step 5), then you should be able to push your code.
"""


"""
Dataset specific notes:

MuchMore Springer Bilingual Corpus

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


"""
Step 2: Create keyword descriptors for your dataset

The following variables are used to populate the dataset entry. Common ones include:

- `_DATASETNAME` = "your_dataset_name"
- `_CITATION`: Latex-style citation of the dataset
- `_DESCRIPTION`: Explanation of the dataset
- `_HOMEPAGE`: Where to find the dataset's hosted location
- `_LICENSE`: License to use the dataset
- `_URLs`: How to download the dataset(s), by name; make this in the form of a dictionary where <dataset_name> is the key and <url_of_dataset> is the value
- `_VERSION`: Version of the dataset
"""

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
# "The cross-lingual information access prototype system for the medical domain will be made publicly accessible through the internet. It provides access to multilingual information on the basis of a domain ontology and classification. For the main task of multilingual domain modelling, the project will focus on German and English. "
_LICENSE = ""

# TODO: there are 4 files in this corpus. we create several configs
# will eventually need to put something like one config per task
# this dataset supports?
_URLs = {
    "en_plain": "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
    "de_plain": "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
    "en_de_plain": [
        "https://muchmore.dfki.de/pubs/springer_english_train_plain.tar.gz",
        "https://muchmore.dfki.de/pubs/springer_german_train_plain.tar.gz",
    ],
    "en_anno": "https://muchmore.dfki.de/pubs/springer_english_train_V4.2.tar.gz",
    "de_anno": "https://muchmore.dfki.de/pubs/springer_german_train_V4.2.tar.gz",
    "muchmore": "https://muchmore.dfki.de/pubs/springer_english_train_V4.2.tar.gz",
}

# took version from annotated file names
_VERSION = "4.2.0"


"""
Step 3: Change the class name to correspond to your <Your_Dataset_Name>
ex: "ChemProtDataset".

Then, fill all relevant information to `BuilderConfig` which populates information about the class. You may have multiple builder configs (ex: a large dataset separated into multiple partitions) if you populate for different dataset names + descriptions. The following is setup for just 1 dataset, but can be adjusted.

NOTE - train/test/dev splits can be handled in `_split_generators`.
"""

NATIVE_ENCODING = "ISO-8859-1"

class MuchMoreDataset(datasets.GeneratorBasedBuilder):
    """MuchMore Springer Bilingual Corpus"""

    VERSION = datasets.Version(_VERSION)
    DEFAULT_CONFIG_NAME = _DATASETNAME

    BUILDER_CONFIGS = [
        # a place for the default config
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
            name="en_de_plain",
            version=VERSION,
            description="muchmore: plaintext of matched english + german abstracts",
        ),
    ]

    # default config produces english annotations at the moment
    def _info(self):

        if self.config.name == self.DEFAULT_CONFIG_NAME:
            features = datasets.Features({
                "sample_id": datasets.Value("string"),
                "corresp": datasets.Value("string"),
                "language": datasets.Value("string"),
                "sentences": datasets.Sequence({
                    "id": datasets.Value("string"),
                    "corresp": datasets.Value("string"),
                    "umlsterms": datasets.Sequence({
                        "id": datasets.Value("string"),
                        "from": datasets.Value("string"),
                        "to": datasets.Value("string"),
                        "concepts": datasets.Sequence({
                            "id": datasets.Value("string"),
                            "cui": datasets.Value("string"),
                            "preferred": datasets.Value("string"),
                            "tui": datasets.Value("string"),
                            "mshs": datasets.Sequence({
                                "code": datasets.Value("string"),
                            }),
                        }),
                    }),
                    "ewnterms": datasets.Sequence({
                        "id": datasets.Value("string"),
                        "to": datasets.Value("string"),
                        "from": datasets.Value("string"),
                        "senses": datasets.Sequence({
                            "offset": datasets.Value("string"),
                        }),
                    }),
                    "semrels": datasets.Sequence({
                        "id": datasets.Value("string"),
                        "term1": datasets.Value("string"),
                        "term2": datasets.Value("string"),
                        "reltype": datasets.Value("string"),
                    }),
                    "chunks": datasets.Sequence({
                        "id": datasets.Value("string"),
                        "to": datasets.Value("string"),
                        "from": datasets.Value("string"),
                        "type": datasets.Value("string"),
                    }),
                    "tokens": datasets.Sequence({
                        "id": datasets.Value("string"),
                        "pos": datasets.Value("string"),
                        "lemma": datasets.Value("string"),
                        "text": datasets.Value("string"),
                    }),
                })
            })

        elif self.config.name in ["en_plain", "de_plain"]:
            features = datasets.Features({
                "sample_id": datasets.Value("string"),
                "sample_id_prefix": datasets.Value("string"),
                "language": datasets.Value("string"),
                "abstract": datasets.Value("string"),
            })

        elif self.config.name == "en_de_plain":
            features = datasets.Features({
                "sample_id_prefix": datasets.Value("string"),
                "sample_id_en": datasets.Value("string"),
                "sample_id_de": datasets.Value("string"),
                "abstract_en": datasets.Value("string"),
                "abstract_de": datasets.Value("string"),
            })



        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
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
                # These kwargs will be passed to _generate_examples
                # `iter_archive` will yield (file_name, file_pointer)
                # tuples for each abstract / member of the tar.gz
                # file when iterated over.
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


    def _generate_examples(self, file_names_and_pointers, split):

        if self.config.name == _DATASETNAME:
            _id = 0
            for file_name, fp in file_names_and_pointers:

                content_bytes = fp.read()
                content_str = content_bytes.decode(NATIVE_ENCODING)
                if content_str == "":
                    print(content_str)
                    print("skipping")
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

                yield _id, {
                    "sample_id": xroot.get("id"),
                    "corresp": xroot.get("corresp"),
                    "language": xroot.get("lang"),
                    "sentences": sentences,
                },
                _id += 1

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

        elif self.config.name == "en_de_plain":
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
                    "sample_id_prefix": sample_id_prefix,
                    "sample_id_en": sample_pair[en_idx]["sample_id"],
                    "sample_id_de": sample_pair[de_idx]["sample_id"],
                    "abstract_en": sample_pair[en_idx]["abstract"],
                    "abstract_de": sample_pair[de_idx]["abstract"],
                }
                _id += 1
