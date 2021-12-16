"""
Ensure your data files are in the proper format.
I.e: https://github.com/lfurrer/bconv
"""
import json
from typing import List, Optional, Dict

import numpy as np
from pybrat.parser import Example, Relation, Entity, BratParser

# import bconv # ??
import bioc
from bioc.bioc import BioCDocument, BioCPassage


# -------------------------- #
# BioC -> BRAT
# -------------------------- #


class BioCXML:
    """
    Creates a
    """

    def __init__(self, kb_key_name: str):
        """
        Instantiates a Parser to translate BioC -> PyBrat style

        :param kb_key_name: Entity <-> Relation identifier for BioC
        """
        self.kb_key_name = kb_key_name

    def parse(self, fname: str) -> List[Example]:
        """
        Construct a pybrat style object from BioC-XML file types

        :param fname: Name of the XML BioC file
        """

        # Read the dataset
        reader = bioc.BioCXMLDocumentReader(fname)
        data = []

        for doc in reader:
            data.append(_get_document_biocxml(doc, self.kb_key_name))

        return data


def _get_document_biocxml(
    doc: BioCDocument,
    kb_key_name: str,
):
    """
    Makes an analog to PyBrat parser "Example"

    Something to note, bioxml can have multiple locations
    for larger words. Seems like Entities have an overall knowledge base identifier

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

    Some entities get 2 identifiers as: <KEY 1>|<KEY 2>; I split this by a bar

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

            for span_key in span.infons[kb_key_name].split("|"):
                ent = Entity(
                    mention=span.text,
                    type=span.infons["type"], # NL type as opposed to DB key
                    start=char_start,
                    end=char_end,
                    id=span.id,
                )

                ent_list.append(ent)

                ents.update({span_key: ent})

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


class PICOParser:
    def __init__(self):
        """
        Instantiates a Parser to load PICO data

        """

    def parse(
        self, data_root: str, sentence_file: str, annotation_files: Dict[str, str]
    ) -> List[Example]:
        """"""
        # load sentences
        _path = data_root / sentence_file
        with open(_path) as fp:
            sentences = json.load(fp)
            # logger.info("Sentences successfully loaded")

        # load annotations
        annotation_dict = {}
        for annotation_type, _file in annotation_files.items():
            _path = data_root / _file.split("/")[-1]
            with open(_path) as fp:
                annotations = json.load(fp)
                annotation_dict[annotation_type] = annotations
        # logger.info("Annotations successfully loaded")

        train = []
        for sentence_id, sentence in sentences.items():

            ents = self._get_entities_pico(annotation_dict, sentence, sentence_id)

            Document = Example(
                text=sentence,
                entities=ents,
                relations=[],
                id=sentence_id,
            )

            train.append(Document)

        return train

    def _get_entities_pico(
        self, annotation_dict: Dict[str, str], sentence: str, sentence_id: int
    ):
        """"""

        def _partition(alist, indices):
            return [alist[i:j] for i, j in zip([0] + indices, indices + [None])]

        ents = []

        for annotation_type, annotations in annotation_dict.items():
            # get indices from three annotators by majority voting
            indices = np.where(
                np.round(np.mean(annotations[sentence_id]["annotations"], axis=0)) == 1
            )[0]

            if len(indices) > 0:  # if annotations exist for this sentence
                split_indices = []
                # if there are two annotations of one type in one sentence
                for item_index, item in enumerate(indices):
                    if item_index + 1 == len(indices):
                        break
                    if indices[item_index] + 1 != indices[item_index + 1]:
                        split_indices.append(item_index + 1)
                multiple_indices = _partition(indices, split_indices)

                for _indices in multiple_indices:

                    annotation_text = " ".join(
                        [sentence.split()[ind] for ind in _indices]
                    )

                    char_start = sentence.find(annotation_text)
                    char_end = char_start + len(annotation_text)

                    ent = Entity(
                        mention=annotation_text,
                        type=annotation_type,
                        start=char_start,
                        end=char_end,
                        id=None,
                    )

                    ents.append(ent)
        return ents
