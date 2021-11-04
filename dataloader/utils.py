"""
Ensure your data files are in the proper format.
I.e: https://github.com/lfurrer/bconv
"""
from typing import List, Optional, Dict
from pybrat.parser import Example, Relation, Entity, BratParser

# import bconv # ??
import bioc
from bioc.bioc import BioCDocument, BioCPassage


def biocxml2brat(fname: str, kb_key_name: str = "MESH") -> List[Example]:
    """
    Converts BioC_XML file to BRAT

    Document <=> Example

    :param fname: name of the file to convert to brat

    :returns brat_path: Location of the brat files
    """

    # Read the dataset
    reader = bioc.BioCXMLDocumentReader(str(fname))
    data = []

    for doc in reader:
        data.append(_get_document_biocxml(doc, kb_key_name))

    return data


def _get_document_biocxml(
    doc: BioCDocument,
    kb_key_name: str,
):
    """
    Makes an analog to PyBrat parser "Example"

    Something to note, bioxml can have multiple locations
    for larger words

    TODO: Do we need events etc??
    """
    text, ent_dict, ents = _get_text_entities_biocxml(doc, kb_key_name)
    relns = _get_relations_biocxml(doc, ent_dict)

    Document = Example(
        text=text,
        entities=ents,
        relations=relns,
        id=doc.id,
    )
    return Document


def _get_text_entities_biocxml(
    doc: BioCDocument,
    kb_key_name: str,
) -> (str, Dict[str, Entity], List[Entity]):
    """
    For a bioC Document, get the entities.
    TODO - issue with span kb_key_name

    How to handle multiple mentions?

    :param doc: Document with passages/annotations
    :param kb_key_name: Name of the knowledge base concept mapping
    """
    text = []
    ents = {}
    ent_list = []

    # Get the entities
    for passage in doc.passages:
        text.append(passage.text)

        for span in passage.annotations:
            char_start = [j.offset for j in span.locations]
            char_end = [j.offset + j.length for j in span.locations]

            # Span is continuous, hence no need for tuple
            if len(char_start) < 2:
                char_start = char_start[0]
                char_end = char_end[0]

            ent = Entity(
                mention=span.text,
                type=span.infons["type"],
                start=char_start,
                end=char_end,
                id=span.id,
            )

            ent_list.append(ent)

            ents.update({span.infons[kb_key_name]: ent})

    return "\n".join(text), ents, ent_list


def _get_relations_biocxml(
    doc: BioCDocument,
    ents: Dict[str, Entity],
) -> List[Relation]:
    """
    For a bioC Document, get the relations.

    ASSUMES sorted dict where first non-info key is ent1, and the other is ent2

    TODO: Is `relation` always the generic type key???
    """
    info_key = "relation"
    relns = []

    for r in doc.relations:
        info = r.infons
        arg1, arg2 = [key for key in info.keys() if key != info_key]
        relns.append(
            Relation(
                type=info[info_key],
                arg1=ents[info[arg1]],
                arg2=ents[info[arg2]],
                id=r.id,
            )
        )

    return relns
