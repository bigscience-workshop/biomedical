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
This dataset contains 225 annotated article titles/abstracts taken from PubMed.
Each entry is annotated in XML format with tagged protein entities, and as such
the dataset is suitable for the task of Named Entity Recognition. 

Dataset was downloaded from https://www.cs.utexas.edu/ftp/mooney/bio-data/proteins.tar.gz
and parsed into the bigbio_kb schema format. Because the orginal paper doesn't
share analysis code, and doesn't list an explicit schema, the schema for
"source" is also parsed into bigbio_kb. 

The dataset was originally collated and analyzed in the following paper:
Comparative experiments on learning information extractors for proteins and their interactions.
Bunescu, Razvan et al. Artificial intelligence in medicine. 33, 2 139-155. 2005.
"""

import os
from glob import glob
import re
from typing import List, Tuple, Dict
from xml.sax.handler import EntityResolver
from xml.etree import ElementTree

from bs4 import BeautifulSoup
import datasets
from yaml import parse
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{bunescu2005comparitive,
  title={Comparative experiments on learning information extractors for proteins and their interactions.},
  author={Bunescu, Razvan et al.},
  journal={Artificial intelligence in medicine},
  volume={33},
  number={2},
  pages={139-155},
  year={2005}
}
"""

_DATASETNAME = "aimed_proteins"

_DESCRIPTION = """\
This dataset is designed for the NLP task of NER.
"""

_HOMEPAGE = ""

_LICENSE = ""  # TODO: NULL

_URLS = {
    _DATASETNAME: ["https://www.cs.utexas.edu/ftp/mooney/bio-data/proteins.tar.gz"],
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


def _get_example_text(example: dict) -> str:
    """
    Concatenate all text from passages in an example of a KB schema
    :param example: An instance of the KB schema
    """
    return " ".join([t for p in example["passages"] for t in p["text"]])


def wrap_xml(xml_text: str) -> str:
    """
    The standard libary xml module requires that xml have some kind of root element.
    This wraps the <ArticleTitle></ArticleTitle><AbstractText></AbstractText> of the
    PubMed abstracts in a <root> element for the etree parser.
    """
    xml_text = f"<root>{xml_text}</root>"
    return xml_text


def remove_question_mark_tags(xml_text):
    """Removes invalid question mark tags, denoted by <?></?>, from XML and returns the modified text"""
    clean_text = re.sub("<\?>(.*?)</\?>", lambda match: match.group(1), xml_text)
    return clean_text


def preprocess_xml(xml_text: str) -> ElementTree:
    """Applies all preprocessing steps to XML text."""
    xml_text = wrap_xml(xml_text)
    xml_text = remove_question_mark_tags(xml_text)
    return xml_text


def textify(t):
    """Recursively iterates through ElementTree nodes and converts to text."""
    s = []
    if t.text:
        s.append(t.text)
    for child in t:
        s.extend(textify(child))
    if t.tail:
        s.append(t.tail)
    return "".join(s)


def process_passage(node, start_idx, _id=0):
    """Converts an ElementTree node representing a article passage into the appropriate schema."""
    if node.tag == "ArticleTitle":
        _type = "title"
    elif node.tag == "AbstractText":
        _type = "abstract"
    else:
        raise ValueError(
            f'node.tag should be one of ["ArticleTitle", "AbstracText"]. Received node.tag == {node.tag}'
        )

    _text = textify(node)
    _offsets = [start_idx, start_idx + len(_text)]
    passage = {
        "id": _id,
        "type": _type,
        "text": [_text],
        "offsets": [_offsets],
    }
    return passage


def get_passages(t):
    """Gets all passages from an ElementTree representing an annotated entry."""
    state = {"current_idx": 0, "str_chunks": [], "passages": []}

    def _textify(t):
        if t.text:
            state["str_chunks"].append(t.text)
            state["current_idx"] += len(t.text)
        for child in t:
            if child.tag in ["ArticleTitle", "AbstractText"]:
                parsed_passage = process_passage(child, start_idx=state["current_idx"])
                state["passages"].append(parsed_passage)
            _textify(child)
        if t.tail:
            state["str_chunks"].append(t.tail)
            state["current_idx"] += len(t.tail)

    _textify(t)
    return state


def textify_prot(t):
    """Handles textifying proteins. Basically just leaves out the tag's tail attribute. """
    child_tags = [c.tag for c in t]
    s = []
    if t.text:
        s.append(t.text)
    for child in t:
        s.extend(textify(child))
    return "".join(s)


def process_entity(node, start_idx, _id=0):
    '''Converts an ElementTree node representing an entity to the appropriate schema.'''
    if node.tag == "prot":
        _type = "protein"
    else:
        raise ValueError(
            f'node.tag should be one of ["prot"]. Received node.tag == {node.tag}'
        )
    _text = textify_prot(node)
    _offsets = [start_idx, start_idx + len(_text)]
    entity = {
        "id": _id,
        "type": _type,
        "text": [_text],
        "offsets": [_offsets],
        "normalized": [],
    }
    return entity


def get_entities(t):
    """Gets all entities from the root ElementTree object."""
    state = {"current_idx": 0, "str_chunks": [], "entities": []}

    def _textify(t):
        if t.text:
            state["str_chunks"].append(t.text)
            state["current_idx"] += len(t.text)
        for child in t:
            if child.tag in ["prot"]:
                parsed_entities = process_entity(child, start_idx=state["current_idx"])
                state["entities"].append(parsed_entities)
            _textify(child)
        if t.tail:
            state["str_chunks"].append(t.tail)
            state["current_idx"] += len(t.tail)

    _textify(t)
    return state


def parse_article(fpath, _id=0, fulltext=False):
    """Loads an entry from a filepath and parses it into the dataset schema."""
    doc_id = re.search("abstract(\d-\d+)", os.path.basename(fpath)).group(1)
    with open(fpath, "r") as f:
        content = f.read()
        xml_text = preprocess_xml(content)
        t = ElementTree.fromstring(xml_text)
    if t:
        entities = get_entities(t)["entities"]
        passages = get_passages(t)["passages"]
        result = {
            "id": _id,
            "document_id": doc_id,
            "passages": passages,
            "entities": entities,
            "events": [],
            "coreferences": [],
            "relations": []
        }
        if fulltext:
            result["fulltext"] = textify(t)
        return result


def fix_ids(entries, test_offsets=True):
    """Resets all ids, passage_ids, and entity_ids so that each is unique."""
    next_id = 0
    for entry in entries:
        entry["id"] = next_id
        next_id += 1
        for entity in entry["entities"]:
            entity["id"] = next_id
            next_id += 1
        for passage in entry["passages"]:
            passage["id"] = next_id
            next_id += 1
    return entries

def fix_offsets(entries):
    '''
    In some cases, entity and passage offsets would be off by one on their starting indices,
    and sometimes their end indices as well. This wasn't happening in my original scratch 
    notebook, and I couldn't diagnose what was happening, but it may be because of the way
    the example text is constructed in the test. As a result, I'm just passing back through the 
    entries and manually checking for those slightly misaligned offsets and correcting them.
    '''
    for entry in entries:
        fulltext = _get_example_text(entry)
        for passage in entry['passages']:
            for ix, offset_text in enumerate(zip(passage['offsets'], passage['text'])):
                offset, text = offset_text
                start, end = offset
                if fulltext[start:end] != text:
                    if fulltext[start - 1 : end-1] == text:
                        passage['offsets'][ix] = [start - 1 , end - 1]
                    elif fulltext[start -1 : end] == text:
                        passage['offset'][ix] = [start -1, end]
        for entity in entry['entities']:
            for ix, offset_text in enumerate(zip(entity['offsets'], entity['text'])):
                offset, text = offset_text
                start, end = offset
                if fulltext[start:end] != text:
                    if fulltext[start - 1 : end-1] == text:
                        entity['offsets'][ix] = [start - 1 , end - 1]
                    elif fulltext[start -1 : end] == text:
                        entity['offset'][ix] = [start -1, end]

                start, end = entity['offsets'][ix]
                if fulltext[start:end] != text:
                    print(len(text))
                    print(len(fulltext[start:end]))
    return entries

class AIMedProteinsDataset(datasets.GeneratorBasedBuilder):
    """
    This dataset contains 225 annotated article titles/abstracts taken from PubMed.
    Each entry is annotated in XML format with tagged protein entities, and as such
    the dataset is suitable for the task of Named Entity Recognition. 

    The dataset was originally collated and analyzed in the following paper:
    Comparative experiments on learning information extractors for proteins and their interactions.
    Bunescu, Razvan et al. Artificial intelligence in medicine. 33, 2 139-155. 2005.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="aimed_proteins_source",
            version=SOURCE_VERSION,
            description="AIMed source schema",
            schema="source",
            subset_id="aimed",
        ),
        BigBioConfig(
            name="aimed_proteins_bigbio_kb",
            version=BIGBIO_VERSION,
            description="AIMed BigBio schema",
            schema="bigbio_kb",
            subset_id="aimed",
        ),
    ]

    DEFAULT_CONFIG_NAME = "aimed_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = schemas.kb_features
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
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
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, data_dir: str, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        fpaths = glob(data_dir[0] + "/*/**")
        # Downloaded data contains a corrupted, duplicate file
        fpaths = [x for x in fpaths if ('abstract' in x) and ("2-14~" not in x)]
        entries = [parse_article(fpath, 0, True) for fpath in fpaths]
        entries = fix_ids(entries)
        entries = fix_entity_offsets(entries)
        for entry in entries:
            if 'fulltext' in entry:
                del entry['fulltext']
    
        if self.config.schema == "source":
            for key, example in enumerate(entries):
                yield key, example
        elif self.config.schema == "bigbio_kb":
            for key, example in enumerate(entries):
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
