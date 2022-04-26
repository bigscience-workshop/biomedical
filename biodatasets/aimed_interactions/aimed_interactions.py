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
from collections import defaultdict
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

_DATASETNAME = "aimed_interactions"

_DESCRIPTION = """\
This dataset is designed for the NLP tasks of Named Entity Recognition and Relation Extraction.
"""

_HOMEPAGE = ""

_LICENSE = ""  # TODO: NULL

_URLS = {
    _DATASETNAME: ["https://www.cs.utexas.edu/ftp/mooney/bio-data/interactions.tar.gz"],
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


NEXT_ID = {'val': 1}

def gen_id():
    new_id = NEXT_ID['val']
    NEXT_ID['val'] += 1
    return new_id

def _get_example_text(example: dict) -> str:
    """
    Concatenate all text from passages in an example of a KB schema
    :param example: An instance of the KB schema
    """
    return " ".join([t for p in example["passages"] for t in p["text"]])


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
                    elif fulltext[start +1 : end + 1] == text:
                        passage['offsets'][ix] = [start +1, end + 1]
                    elif fulltext[start +1 : end] == text:
                        passage['offsets'][ix] = [start +1, end]
                    elif fulltext[start + 2 : end + 2] == text:
                        passage['offsets'][ix] = [start +2, end+2]
        for entity in entry['entities']:
            for ix, offset_text in enumerate(zip(entity['offsets'], entity['text'])):
                offset, text = offset_text
                start, end = offset
                if fulltext[start:end] != text:
                    if fulltext[start - 1 : end-1] == text:
                        entity['offsets'][ix] = [start - 1 , end - 1]
                    elif fulltext[start -1 : end] == text:
                        entity['offset'][ix] = [start -1, end]
                    elif fulltext[start +1 : end + 1] == text:
                        entity['offsets'][ix] = [start +1, end + 1]
                    elif fulltext[start +1 : end] == text:
                        entity['offsets'][ix] = [start +1, end]

                start, end = entity['offsets'][ix]
                if fulltext[start:end] != text:
                    print(len(text))
                    print(len(fulltext[start:end]))
    return entries

def load(fpath):
    with open(fpath, 'r') as f:
        content = f.read()
    return content

def wrap_xml(xml_text: str, tag='root') -> str:
    '''
    The standard libary xml module requires that xml have some kind of root element. 
    This wraps the <ArticleTitle></ArticleTitle><AbstractText></AbstractText> of the
    PubMed abstracts in a <root> element for the etree parser.
    '''
    xml_text = f"<{tag}>{xml_text}</{tag}>"
    return xml_text

def passages_to_xml(text):
    ti_ix = text.find('TI - ')
    ab_ix = text.find('AB - ')
    ad_ix = text.rfind('AD -')

    ti_slice = text[ti_ix: ab_ix]
    ab_slice = text[ab_ix: ad_ix]
    ad_slice = text[ad_ix:]

    ti_node = wrap_xml(ti_slice, 'ArticleTitle')
    ab_node = wrap_xml(ab_slice, 'AbstractText')
    ad_node = wrap_xml(ad_slice, 'ADText')

    xml_text = wrap_xml(f"{ti_node}{ab_node}{ad_node}")
    return xml_text

def fix_prot_pairs(xml_text):
    return re.sub('<p(\d)  pair=(\d+) >', lambda match: f'<p{match.group(1)}  pair="{match.group(2)}">', xml_text)

def fix_whitespace(xml_text):
    '''In the dataset, all XML tags are followed by two whitespace characters. This removes those two characters.'''
    return re.sub('(<.*?>)  ', lambda match: match.group(1), xml_text)

def preprocess_xml_text(text):
    xml_text = fix_prot_pairs(text)
    xml_text = passages_to_xml(xml_text)
    xml_text = fix_whitespace(xml_text)
    return xml_text

def loadtree(fpath):
    text = load(fpath)
    xml_text = preprocess_xml_text(text)
    tree = ElementTree.fromstring(xml_text)
    return tree

def textify(t):
    s = []
    if t.text:
        s.append(t.text)
    for child in t:
        s.extend(textify(child))
    if t.tail:
        s.append(t.tail)
    return ''.join(s)


def process_passage(node, start_idx):

    if node.tag == 'ArticleTitle':
        _type = 'title'
    elif node.tag == 'AbstractText':
        _type = 'abstract'
    elif node.tag == 'ADText':
        _type = 'ad'
    else:
        raise ValueError(f'node.tag should be one of ["ArticleTitle", "AbstracText", "ADText"]. Received node.tag == {node.tag}')

    _text = textify(node)
    _offsets = [start_idx, start_idx + len(_text)]

    passage = {
        'id': gen_id(),
        'type': _type,
        'text': [_text],
        'offsets': [_offsets]
    }
    return passage

def get_passages(t):
    
    state = {
        'current_idx': 0,
        'str_chunks': [],
        'passages': []
    }
    
    def _textify(t):
        if t.text:
            state['str_chunks'].append(t.text)
            state['current_idx'] += len(t.text)
        for child in t:
            if child.tag in ['ArticleTitle', 'AbstractText', "ADText"]:
                parsed_passage = process_passage(child, start_idx=state['current_idx'])
                state['passages'].append(parsed_passage)
            _textify(child)
        if t.tail:
            state['str_chunks'].append(t.tail)
            state['current_idx'] += len(t.tail)
            
    _textify(t)
    return state 


def textify_prot(t):
    child_tags = [c.tag for c in t]

    s = []
    if t.text:
        s.append(t.text)
    for child in t:
        s.extend(textify(child))
        
    return ''.join(s)


def process_entity(node, start_idx):

    if not hasattr(node, 'tag'):
        return 
    
    if node.tag == 'prot':
        _type = 'protein'
    else:
        raise ValueError(f'node.tag should be one of ["prot"]. Received node.tag == {node.tag}')

        
    _text = textify_prot(node)
    _offsets = [start_idx, start_idx + len(_text)]

    entity = {
        'id': gen_id(),
        'type': _type,
        'text': [_text],
        'offsets': [_offsets],
        'normalized': []
    }
    return entity


def get_entities_and_relations(t):
    
    state = {
        'current_idx': 0,
        'str_chunks': [],
        'entities': [],
        'pairs': defaultdict(lambda: {"id": gen_id(), "type": "protein-protein interaction", "normalized": []})
    }
    
    def _textify(t, parent_tag='root'):
        if t.text:
            state['str_chunks'].append(t.text)
            state['current_idx'] += len(t.text)
        for child in t:
            if child.tag in ['p1', 'p2']:
                n = child.tag[1]
                pair_num = child.get('pair')
                prot = child.find('prot')
                if prot:
                    entity = process_entity(prot, start_idx=state['current_idx'])
                    state['pairs'][pair_num][f'arg{n}_id'] = entity['id']
                    state['entities'].append(entity)

            if child.tag in ['prot'] and parent_tag not in ['p1', 'p2']:
                parsed_entity = process_entity(child, start_idx=state['current_idx'])
                state['entities'].append(parsed_entity)
                
            _textify(child, parent_tag=t.tag)
        if t.tail:
            state['str_chunks'].append(t.tail)
            state['current_idx'] += len(t.tail)
            
    _textify(t)
    relations = []
    for relation in state['pairs'].values():
        if 'arg1_id' in relation and 'arg2_id' in relation:
            relations.append(relation)

    state['relations'] = relations
    del state['pairs']
    return state 

def parse_article(fpath, fulltext=False):
    
    doc_id = re.search('_.*?(\d+)', fpath).group(1)

    text = load(fpath)
    xml_text = preprocess_xml_text(text)
    t = ElementTree.fromstring(xml_text)
    
    entities_relations = get_entities_and_relations(t)
    entities = entities_relations['entities']
    relations = entities_relations['relations']
    passages = get_passages(t)['passages']
    result =  {
        "id": gen_id(),
        "document_id": doc_id,
        'passages': passages,
        'entities': entities,
        'events': [],
        'coreferences': [],
        'relations': relations
    }

    if fulltext:
        fulltext = textify(t)
        result['fulltext'] = fulltext

    return result


class AIMedInteractionsDataset(datasets.GeneratorBasedBuilder):
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
            name="aimed_interactions_source",
            version=SOURCE_VERSION,
            description="AIMedInteractions source schema",
            schema="source",
            subset_id="aimed_interactions",
        ),
        BigBioConfig(
            name="aimed_interactions_bigbio_kb",
            version=BIGBIO_VERSION,
            description="AIMedInteractions BigBio schema",
            schema="bigbio_kb",
            subset_id="aimed_interactions",
        ),
    ]

    DEFAULT_CONFIG_NAME = "aimed_interactions_source"

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
        fpaths = [x for x in fpaths if ('abstract' in x)]
        
        entries = [parse_article(p, False) for p in fpaths]
        entries = fix_offsets(entries)
    
        if self.config.schema == "source":
            for key, example in enumerate(entries):
                yield key, example
        elif self.config.schema == "bigbio_kb":
            for key, example in enumerate(entries):
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
