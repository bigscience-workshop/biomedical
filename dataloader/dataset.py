"""
Creates a pybrat parser object from an input dataset. The following occurs:

- Given a dataset, if it's not in brat form, converts it to brat
- if data not in brat form, runs `pybrat` parser on it
- for each document, returns Examples

If your data is in brat format, ensure the splits refer to the folder that contains the ".ann" and ".txt" indications.

If your data is in any other form, you will need an extension (i.e. .xml)

Example:
    Example.text = <input text data>
    
    Example.entities = List[Entities]
        for a given "i":
        entities[i].type = <label/type of the entity>
        entities[i].mention = <text corresponding to entity>
        entities[i].start = <start character in overall text>
        entities[i].end = <end character in overall text>
        entities[i].id = <corresponding entity id>

    Example.relations = List[Relations]
        for a given "i":
            relations[i].type = <label/type of entity>
            relations[i].arg1 = Entity <first entity in reln>
            relations[i].arg2 = Entity <second entity in reln>
            relations[i].id = <corresponding entity id>
    
    Example.events = List[Events] (TODO)
"""
import os
from loguru import logger
from pybrat.parser import Example, Relation, Entity, BratParser

from typing import Dict, Callable, List, Optional
from utils import biocxml2brat

parser_lookup = {
    "bioc_xml": biocxml2brat,
    "bioc_json": None,
    "brat": BratParser(error="ignore"),
}


class Dataset:
    """
    Given a Biomedical NLP dataset, converts a series of documents (entities, events, relations) in the pybrat parser format

    :param data_root: Head directory
    :param split_names: Subfolders/filenames with data splits (ex: data_root/train/)
    :param fmt: Data-type original format
    """

    def __init__(
        self,
        data_root: str,
        split_names: Dict[str, str],
        fmt: str,
        parser: Optional[BratParser] = None,
        kb_key_name: Optional[str] = None,
    ):
        """
        Creates a dataset in the pybrat parser form

        :param data_root: Location of the brat (ann + text) files
        :param splits: Data split + filename of split
        :param fmt: Type of data file
        :param parser: Type of parser (BratParser)
        """

        self.data_root = data_root
        self.split_names = split_names
        self.format = fmt.lower()
        self.kb_key_name = kb_key_name # Only for BioC-XML

        # Initialize the parser
        self._init_parser(parser)

        logger.info(f"Number of datasplits= {len(self.split_names)}")
        self.data = {split: None for split in self.split_names.keys()}

        # For each data split, get the path where brat files live
        for split, fname in self.split_names.items():
            dtype_path = os.path.join(self.data_root, fname)

            self.data[split] = self.parse(dtype_path, self.format)

    def parse(self, fname: str, fmt: str) -> List[Example]:
        """
        Calls the brat parser on brat dataset

        :param fname: path/datafile to parse
        :param fmt: data type format

        """
        logger.info(f"Parsing file {fname}")

        if self.parser is not None:
            return self.parser.parse(fname)
        else:
            return None
         

    def _init_parser(self, parser: Optional[BratParser]) -> None:
        """Initializes a parser if None is provided"""
        if parser is None:
            logger.info(f"No parser provided, using default")
            self.parser = parser_lookup[self.format]
        else:
            self.parser = parser

