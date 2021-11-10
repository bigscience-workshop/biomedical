"""
Note-BioC may have overlaps in start/end positions

Creates a pybrat parser object from an input dataset. The following occurs:

- Given a dataset, if it's not in brat form, converts it to brat
- if data not in brat form, runs `pybrat` parser on it
- for each document, returns Examples

If your data is in brat format, ensure the splits refer to the folder that contains the ".ann" and ".txt" indications.

If you specify a dataset that has an existing DataLoader, then the format is assumed 'brat' even if the original source is not. This is because the data is downloaded and processed in the brat form automatically.

If your data is in any other form, you will need to specify the extension via fmt; if a parser exists, this data will be processed.

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

TODOs:
    - in dataloaders, probably want to remove the tmp/ directory created
    - make uncompress function more modular (gzip/tar)
    - currently doesn't handle if data is partially downloaded; these are small files though
"""
import os
from loguru import logger
from pybrat.parser import Example, Relation, Entity, BratParser

from typing import Dict, Callable, List, Optional
from .utils.formatters import BioCXML
from .utils.dataloaders import JNLPBA, CellFinder, Linneaus, DDI, ChemProt, BC5CDR

parser_lookup = {
    "bioc_xml": BioCXML,
    "bioc_json": None,
    "brat": BratParser(error="ignore"),
    "conll": None,
}

dataloader_lookup = {
    "jnlpba": JNLPBA,
    "cellfinder": CellFinder,
    "linneaus": Linneaus,
    "chemprot": ChemProt,
    "ddi": DDI,
    "bc5cdr": BC5CDR,
}

parser_overrides = {
    "bc5cdr": BioCXML(kb_key_name="MESH"),
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
        dataset: str,
        data_root: str = ".",
        fmt: str = "brat",
        split_names: Optional[Dict[str, str]] = None,
        parser: Optional[BratParser] = None,
        # kb_key_name: Optional[str] = None,
    ):
        """
        Creates a dataset in the pybrat parser form

        :param dataset: Name of the dataset; if supported, downloads and prepares data in pre-made format
        :param data_root: Path to the dataset
        :param fmt: Type of data file
        :param splits: Data split + filename of split
        :param parser: Type of parser (BratParser)

        Further kwargs are for the parser instantiation
        """

        self.dataset = dataset.lower()
        self.data_root = data_root
        self.format = fmt.lower()

        # self.kb_key_name = kb_key_name # Only for BioC-XML
        self.split_names = split_names

        # Initialize the parser
        self._init_parser(parser)

        # Lookup dataset
        if self.dataset in dataloader_lookup.keys():
            logger.info("Dataset=" + self.dataset + ".")
            self.load_and_parse()
        else:
            logger.info("Custom dataset specified named=" + self.dataset)
            raise NotImplementedError(f"NOT IMPLEMENTED YET")

    def load_and_parse(self):
        """
        For a dataset with a reserved dataloader class, download and process the files for use.

        Each dataset's splits can be accessed as such: `self.data.train`.
        For more details for each dataloader, see `utils_dataloaders`
        """
        self.data = dataloader_lookup[self.dataset](self.data_root, self.parser)

    def _init_parser(self, parser: Optional[BratParser], **kwargs):
        """
        Initializes a pybrat parser if None is provided
        Subsequent kwargs are passed to instantiat the parser

        TODO: This isn't great :/
        """
        if parser is None:
            logger.info(f"No parser provided, using default for file/data type")
            if self.dataset not in parser_overrides.keys():
                self.parser = parser_lookup[self.format]
            else:
                self.parser = parser_overrides[self.dataset]
        else:
            logger.info("User provided parser.")
            self.parser = parser(**kwargs)
