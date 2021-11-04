"""
Creates a pybrat parser object from an input dataset. The following occurs:

- Given a dataset, if it's not in brat form, converts it to brat
- if data not in brat form, runs `pybrat` parser on it
- for each document, returns Examples

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
from utils import biocxml2brat, biocjson2brat

fmt_lookup = {
    "bioc_xml": biocxml2brat,
    "bioc_json": biocjson2brat,
    "brat": None,
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
        self.orig_fmt = fmt.lower()

        # Initialize the parser
        self._init_parser(parser)

        logger.info(f"Number of datasplits= {len(self.split_names)}")
        self.data = {split: None for split in self.split_names.keys()}

        # For each data split, get the path where brat files live
        for split, fname in self.split_names.items():
            dtype_path = os.path.join(self.data_root, fname)

            self.data[split] = self.parse(dtype_path, self.orig_fmt)

    def parse(self, fname: str, fmt: str) -> List[Example]:
        """
        Calls the brat parser on brat dataset

        :param fname: path/datafile to parse
        :param fmt: data type format

        """
        brat_path = self._convert2brat(fname, fmt_lookup, fmt)

        logger.info(f"Parsing file {fname}")

        return self.parser.parse(fname)

    def _init_parser(self, parser: Optional[BratParser]) -> None:
        """Initializes a parser if None is provided"""
        if parser is None:
            logger.info(f"No parser provided, using default")
            self.parser = BratParser(error="ignore")
        else:
            self.parser = parser

    @staticmethod
    def _convert2brat(
        dtype_path: str,
        fmt_lookup: Dict[str, Callable[[str], str]],
        fmt: str,
    ) -> str:
        """
        Converts dataset to brat format if in non-brat form

        :param dtype_path: location of the data split files
        :param fmt_lookup: Dictionary for data type format to brat converter
        :param fmt: type of data format

        :returns: path for the brat files

        """
        convert_fxn = fmt_lookup.get(fmt, None)

        if convert_fxn is not None:
            brat_path = convert_fxn(dtype_path)
        else:
            brat_path = dtype_path

        return brat_path
