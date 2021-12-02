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
from functools import partial

from loguru import logger
from pybrat.parser import Example, Relation, Entity, BratParser
from flair.data import Sentence

import copy
from typing import Dict, Callable, List, Optional
from dataloader.utils.formatters import BioCXML
from dataloader.utils.dataloaders import JNLPBA, CellFinder, Linneaus, DDI, ChemProt, \
    BC5CDR, HunFlairDataset

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
    # CELLLINE
    "hunflair_cellline_cll": partial(HunFlairDataset, name="hunflair_cellline_cll"),
    "hunflair_cellline_cellfinder": partial(HunFlairDataset,
                                            name="hunflair_cellline_cellfinder"),
    "hunflair_cellline_gellus": partial(HunFlairDataset,
                                        name="hunflair_cellline_gellus"),
    "hunflair_cellline_jnlpba": partial(HunFlairDataset,
                                        name="hunflair_cellline_jnlpba"),
    # CHEMICAL
    "hunflair_chemical_chebi": partial(HunFlairDataset, name="hunflair_chemical_chebi"),
    "hunflair_chemical_cdr": partial(HunFlairDataset, name="hunflair_chemical_cdr"),
    "hunflair_chemical_cemp": partial(HunFlairDataset, name="hunflair_chemical_cemp"),
    "hunflair_chemical_scai": partial(HunFlairDataset, name="hunflair_chemical_scai"),
    # OFFLINE "hunflair_chemical_bionlp2013_cg": partial(HunFlairDataset, name=# OFFLINE "hunflair_chemical_bionlp2013_cg"),
    # DISEASE
    "hunflair_disease_cdr": partial(HunFlairDataset, name="hunflair_disease_cdr"),
    "hunflair_disease_pdr": partial(HunFlairDataset, name="hunflair_disease_pdr"),
    "hunflair_disease_ncbi": partial(HunFlairDataset, name="hunflair_disease_ncbi"),
    "hunflair_disease_scai": partial(HunFlairDataset, name="hunflair_disease_scai"),
    "hunflair_disease_mirna": partial(HunFlairDataset, name="hunflair_disease_mirna"),
    # OFFLINE "hunflair_disease_bionlp2013_cg" : partial(HunFlairDataset, name=#OFFLINE "hunflair_disease_bionlp2013_cg" ),
    "hunflair_disease_variome": partial(HunFlairDataset,
                                        name="hunflair_disease_variome"),
    # GENE/PROTEIN
    "hunflair_gene_variome": partial(HunFlairDataset, name="hunflair_gene_variome"),
    "hunflair_gene_fsu": partial(HunFlairDataset, name="hunflair_gene_fsu"),
    "hunflair_gene_gpro": partial(HunFlairDataset, name="hunflair_gene_gpro"),
    "hunflair_gene_deca": partial(HunFlairDataset, name="hunflair_gene_deca"),
    "hunflair_gene_iepa": partial(HunFlairDataset, name="hunflair_gene_iepa"),
    "hunflair_gene_bc2gm": partial(HunFlairDataset, name="hunflair_gene_bc2gm"),
    "hunflair_gene_bio_infer": partial(HunFlairDataset, name="hunflair_gene_bio_infer"),
    # OFFLINE "hunflair_gene_bionlp2013_cg": partial(HunFlairDataset, name=#OFFLINE "hunflair_gene_bionlp2013_cg"),
    "hunflair_gene_cellfinder": partial(HunFlairDataset,
                                        name="hunflair_gene_cellfinder"),
    "hunflair_gene_chebi": partial(HunFlairDataset, name="hunflair_gene_chebi"),
    "hunflair_gene_craftv4": partial(HunFlairDataset, name="hunflair_gene_craftv4"),
    "hunflair_gene_jnlpba": partial(HunFlairDataset, name="hunflair_gene_jnlpba"),
    "hunflair_gene_loctext": partial(HunFlairDataset, name="hunflair_gene_loctext"),
    "hunflair_gene_mirna": partial(HunFlairDataset, name="hunflair_gene_mirna"),
    "hunflair_gene_osiris": partial(HunFlairDataset, name="hunflair_gene_osiris"),
    # SPECIES
    "hunflair_species_cellfinder": partial(HunFlairDataset,
                                           name="hunflair_species_cellfinder"),
    "hunflair_species_s800": partial(HunFlairDataset, name="hunflair_species_s800"),
    "hunflair_species_chebi": partial(HunFlairDataset, name="hunflair_species_chebi"),
    "hunflair_species_mirna": partial(HunFlairDataset, name="hunflair_species_mirna"),
    "hunflair_species_loctext": partial(HunFlairDataset,
                                        name="hunflair_species_loctext"),
    # OFFLINE "hunflair_species_bionlp2013_cg": partial(HunFlairDataset, name=# OFFLINE "hunflair_species_bionlp2013_cg"),
    "hunflair_species_craftv4": partial(HunFlairDataset,
                                        name="hunflair_species_craftv4"),
    "hunflair_species_linneaus": partial(HunFlairDataset,
                                         name="hunflair_species_linneaus"),
    "hunflair_species_variome": partial(HunFlairDataset,
                                        name="hunflair_species_variome"),
}

parser_overrides = {
    "bc5cdr": BioCXML(kb_key_name="MESH"),
}


class BioDataset:
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
        self.split_names = split_names  # Train/Dev/Test custom datasets

        # Initialize the parser
        self._init_parser(parser)

        # Lookup dataset
        if self.dataset in dataloader_lookup.keys():
            logger.info("Dataset=" + self.dataset + ".")
            self.load_and_parse()
        else:
            logger.info("Custom dataset specified named=" + self.dataset)
            raise NotImplementedError(f"No custom support yet!")

        # load entities and relations type
        self.natural_entity_types = dict()
        self.natural_relation_types = dict()
        for instance in self.data.train:
            entities = instance.entities
            relations = instance.relations

            for entity in entities:
                self.natural_entity_types[entity.type] = entity.type.lower()

            for relation in relations:
                self.natural_relation_types[relation.type] = relation.type.lower()

        # convert character idx to token idx
        if self.dataset not in ["linneaus", "bc5cdr", "cellfinder"]:
            logger.info("Dataset processing for TANL")
            self.data.train = self.convert_for_tanl(self.data.train)
            self.data.test = self.convert_for_tanl(self.data.test)

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

    def convert_for_tanl(self, data):
        """
        Convert data into TANL-compatible format.
        - use tokens instead of text.
        - entities: include tanl_start and tanl_end, both of which points to the token index instead of character index.
        - relations: include tanl_head and tanl_tail, both of which points to the entity index.
            For instance, instance.entities[instance.relations[0].tanl_head] would give the head entity
            for the first relation
        """
        for instance in data:
            sentence = Sentence(instance.text)
            instance.tokens = [token.text for token in sentence]

        data = self.convert_entities(data)
        data = self.convert_relations(data)
        return data

    def convert_entities(self, data):
        """
        include tanl_start and tanl_end, both of which points to the token index instead of character index.
        """
        new_instances = list()
        for j, instance in enumerate(data):
            new_entities = copy.deepcopy(instance.entities)
            if not new_entities:
                new_instances.append(instance)
                continue

            sentence = Sentence(instance.text)
            for ptr in range(len(new_entities)):
                start_idx = -1
                flag = False
                for token_idx, token in enumerate(sentence):
                    if new_entities[ptr].start <= token.end_pos and token.start_pos < \
                            new_entities[ptr].end:
                        if start_idx == -1:
                            start_idx = token_idx

                    if start_idx != -1 and token.start_pos >= new_entities[ptr].end:
                        new_entities[ptr].tanl_start = start_idx
                        new_entities[ptr].tanl_end = token_idx
                        break
                else:
                    if start_idx != -1 and sentence[-1].end_pos == new_entities[ptr].end:
                        new_entities[ptr].tanl_start = start_idx
                        new_entities[ptr].tanl_end = len(sentence)
                    else:
                        flag = True

            if not flag:
                instance.entities = new_entities
                new_instances.append(instance)
            else:
                print("instance malformed:", instance)
        return new_instances

    def convert_relations(self, data):
        """
        include tanl_head and tanl_tail, both of which points to the entity index.
            For instance, instance.entities[instance.relations[0].tanl_head] would give the head entity
            for the first relation
        """
        new_instances = list()
        for j, instance in enumerate(data):
            new_relations = copy.deepcopy(instance.relations)
            if not new_relations:
                new_instances.append(instance)
                continue

            map_idx_to_entity = dict()
            for entity_idx, entity in enumerate(instance.entities):
                map_idx_to_entity[(entity.start, entity.end)] = entity_idx

            sentence = Sentence(instance.text)
            for ptr in range(len(new_relations)):
                flag = False
                if new_relations[ptr].arg1 is None or new_relations[ptr].arg2 is None:
                    print("arg1 or arg2 is None:", instance)
                    flag = True
                    break

                # head and tail points to the index of entity in the list instance.entities
                new_relations[ptr].tanl_head = map_idx_to_entity[
                    (new_relations[ptr].arg1.start, new_relations[ptr].arg1.end)]
                new_relations[ptr].tanl_tail = map_idx_to_entity[
                    (new_relations[ptr].arg2.start, new_relations[ptr].arg2.end)]

            if not flag:
                instance.relations = new_relations
                new_instances.append(instance)
        return new_instances
