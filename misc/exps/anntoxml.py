#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

# Preamble {{{
from __future__ import with_statement
import cgi

try:
    import annotation
except ImportError:
    import os.path
    from sys import path as sys_path
    # Guessing that we might be in the brat tools/ directory ...
    sys_path.append(os.path.join(os.path.dirname(__file__), '../server/src'))
    import annotation

# this seems to be necessary for annotations to find its config
sys_path.append(os.path.join(os.path.dirname(__file__), '..'))
# }}}



# Processing {{{
def start_tag(entity):
    attrlist = [entity.type]
    for attribute in entity.attributes:
        value = attribute.value
        if isinstance(value, bool):
            value = str(value).lower()
        attrlist.append(u'%s="%s"' % (attribute.type, cgi.escape(value)))
    return u'<%s>' % ' '.join(attrlist)

def end_tag(entity):
    return u'</%s>' % entity.type

def convert_files(docname, root, txtname, out):
    ann = annotation.TextAnnotations(docname)
    with open(txtname) as r:
        txt = r.read().decode('utf8')
    entities = list(ann.get_entities())
    for entity in entities:
        entity.attributes = []
    attributes = list(ann.get_attributes())
    entity_dict = {entity.id: entity for entity in entities}
    for attribute in attributes:
        try:
            entity = entity_dict[attribute.target]
            entity.attributes.append(attribute)
        except KeyError:
            # ignore event attributes
            pass
    startlist = [(entity.spans[0][0], -entity.spans[0][1], False, index, start_tag(entity)) for index, entity in enumerate(entities)]
    endlist = [(entity.spans[0][1], -entity.spans[0][0], True, -index, end_tag(entity)) for index, entity in enumerate(entities)]
    lastpos = len(txt)
    xml = ""
    for pos, _, _, _, tag in sorted(startlist + endlist, reverse=True):
        xml = tag + cgi.escape(txt[pos:lastpos]) + xml
        lastpos = pos
    xml = u'<%s>%s</%s>\n' % (root, cgi.escape(txt[0:lastpos]) + xml, root)
    out.write(xml.encode('utf8'))
# }}}



# Parsing command line {{{
KNOWN_FILE_SUFF = [annotation.TEXT_FILE_SUFFIX] + annotation.KNOWN_FILE_SUFF
EXTENSIONS_RE = '\\.(%s)$' % '|'.join(KNOWN_FILE_SUFF)
def name_without_extension(file_name):
    import re
    return re.sub(EXTENSIONS_RE, '', file_name)

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('annfile', help='The annotation file')
    parser.add_argument('txtfile', nargs='?', default=None, help='The text file')
    parser.add_argument('-r', '--root', default='document', help='The document root')
    parser.add_argument('-o', '--out', default='-', help='The output file')
    opts = parser.parse_args()

    opts.annfile = name_without_extension(opts.annfile)
    opts.root = opts.root.decode('utf8')

    if opts.txtfile is None:
        opts.txtfile = '%s.txt' % opts.annfile

    if opts.out == '-':
        out = sys.stdout
    else:
        out = open(opts.out, 'w')

    convert_files(opts.annfile, opts.root, opts.txtfile, out)

    if opts.out != '-':
        out.close()
# }}}
