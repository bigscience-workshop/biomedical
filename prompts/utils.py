import io
import os
import re
import gzip
import tarfile
import zipfile
import urllib.request
import bioc
import itertools
import collections


def download(url, fpath):
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, fpath)

def uncompress(fpath, outfpath):
    ext = os.path.os.path.splitext(fpath)
    if ext[-1] == ".zip":
        with zipfile.ZipFile(fpath, 'r') as zip_ref:
            # HACK for files containing MacOS garbage files
            for zobj in zip_ref.namelist():
                if '__MACOSX' in zobj:
                    continue
                zip_ref.extract(zobj, path=outfpath)
            #zip_ref.extractall(outfpath)

#
# Some simple BioC tools
# http://bioc.sourceforge.net
#

class Annotation:
    def __init__(self, offsets, text, type_, kb_id):
        self.offsets = offsets
        self.text = text
        self.type_ = type_
        self.kb_id = kb_id

class Document:
    def __init__(self, text, ents, relations):
        self.text = text
        self.ents = ents
        self.relations = relations

def parse_bioc_annotations(xdoc, kb_id_key=None):
    """ Lightweight container object for parsed annotations.

    :param xdoc:
    :param kb_id_key:
    :return:
    """
    text = ' '.join([section.text for section in xdoc.passages])
    ents = []
    for section in xdoc.passages:
        for span in section.annotations:
            char_start = span.locations[0].offset
            char_end = span.locations[0].offset + span.locations[0].length
            offsets = (char_start, char_end)
            kb_id = span.infons[kb_id_key] if kb_id_key else -1
            ents.append(Annotation(offsets, span.text, span.infons['type'], kb_id))

    return Document(text,
                    sorted(ents, key=lambda x:x.offsets, reverse=False),
                    [rel.infons for rel in xdoc.relations])


def load_bioc_corpus(fname, kb_key_name=None):
    """ Load BioC file

    :param fname:
    :param kb_key_name: Concept mappings to a knowledge base (e.g, MESH)
    :return:
    """
    # BioCXMLDocumentReader only supports str paths
    reader = bioc.BioCXMLDocumentReader(str(fname))
    return [
        parse_bioc_annotations(xdoc, kb_key_name)
        for i,xdoc in enumerate(reader)
    ]


#
# Dataset Task Transformations
#

def create_yes_no_choices_from_relations(x, rela_types):
    """Ugly code! Assumes binary relations"""
    # build mapping of MESH ids to surface forms
    kb = {}
    for (a,b) in rela_types:
        kb[a] = collections.defaultdict(set)
        kb[b] = collections.defaultdict(set)
    for e in x.ents:
        kb[e.type_][e.kb_id].add(e.text)
    kb = {type_:dict(kb[type_]) for type_ in kb}

    # sample pos/neg pairs by relation type
    relations = {}
    for a,b in rela_types:
        pos = [(rela[a], rela[b]) for rela in x.relations]
        neg = [r for r in list(itertools.product(kb[a], kb[b])) if r not in pos]
        relations[(a,b)] = {'pos':pos, 'neg':neg}

    # map KB_IDs to surface forms
    for a,b in relations:
        for bucket in ['pos','neg']:
            str_pairs = []
            for a_kb_id, b_kb_id in relations[(a,b)][bucket]:
                # TODO: Bug here with unmapped MESH IDs
                try:
                   a_strs, b_strs = set(kb[a][a_kb_id]), set(kb[b][b_kb_id])
                   str_pairs.append(list(itertools.product(a_strs, b_strs)))
                except Exception as e:
                    print('error creating surface forms', e)
            relations[(a,b)][bucket] = str_pairs

    return relations

def create_multiple_choice_from_relations(x):
    pass