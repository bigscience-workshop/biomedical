class Annotation:
    def __init__(self, offsets, text, type_, kb_id, span):
        self.offsets = offsets
        self.text = text
        self.type_ = type_
        self.kb_id = kb_id
        self.span = span

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
            ents.append(Annotation(offsets, span.text, span.infons['type'], kb_id, span))

    return Document(text, sorted(ents, key=lambda x:x.offsets, reverse=False), [rel.infons for rel in xdoc.relations])


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
