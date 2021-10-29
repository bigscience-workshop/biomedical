"""
Date: 2021.10.10

ChemProt Corpus (https://biocreative.bioinformatics.udel.edu/resources/corpora/chemprot-corpus-biocreative-vi/) prompting annotation

This script does the following:

(1) If the ChemProt corpus is unavailable/unspecified - will download it
(2) Converts the ChemProt corpus into the BRAT/Standoff format to use pybrat
(3) Creates the appropriate Example/Relations
(4) Creates prompts

More details about the dataset can be found in the readme's in the respective datasets.

Overall:

Entities include:
Chemicals: Chemical entities
Gene-Y: Gene/protein annotation with a known biological identifier in a DB (as of 2017)
Gene-N: Gene/protein annotation without a known biological identifier in a database (as of 2017)

Relations Include (exactly quoted from the Readme):

CPR-3: Upregulator, activator, indirect upregulator
CPR-4: downregulator, inhibitor, indirect downregulator
CPR-5: agonist, agonist-activator, agonist inhibitor
CPR-6: antagonist
CPR-9: sybstrate, product of, substrate product of

**NOTE, the implementation here uses the GOLD STANDARD, as this is canonically evaluated against. There are relations outside these annotations provided but were not group-evaluated (ex: CPR:1, CPR:2, CPR:7, CPR:8, and CPR:10 labels)

Some notes:
i) The ChemProt dataset is a zipped folder of zipped folders. The individual train/test/sample/dev folders must be also unzipped

ii) The ChemProt dataset needs to be first compiled into standoff format, thus utils_chemprot handles the TSV -> standoff conversion.

Call this script from the head of the biomedical directory

Usage:

python chemprot.py --outdir your/output/dir
python chemprot.py --outdir your/output/dir --datadir your/chemprot/corpus

"""

import os
import argparse
from pathlib import Path
from loguru import logger
from functools import partial
from typing import List, Callable
from pybrat.parser import BratParser, Example, Relation
from utils import download, uncompress
from prompts import DatasetPrompts

from utils_chemprot import chemprot_2_standoff

# Some prompts can be borrowed for template
from ddi import (
    list_entity_template,
    list_comma_separated_entity_mentions,
    bulleted_list_entity_mentions,
    list_relation_template,
    list_comma_separated_relations,
    bulleted_list_relations,
)

# TODO: Implement proper Hugging Face's Datasets version of ChemProt??

# URL with ChemProt Corpus Data
_URL = "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip"

# Train/Dev/Test dataset prefix; additional "sample" file for quick tests
_TRAIN = "chemprot_training"
_DEV = "chemprot_development"
_TEST = "chemprot_test_gs"
_SAMPLE = "chemprot_sample"

# Standoff + Brat parser
_BRAT_PARSER = BratParser(error="ignore")


def load_chemprot(data_root: str, split: str) -> List[Example]:
    """
    Load the data split (ex: train/test/dev) for ChemProt.
    If BRAT annotation not provided, will create it and process it.

    :param data_root: Location of the ChemProt data directory
    :param split: Name of the data split (train/test/dev/sample)
    """
    data_dir = os.path.join(data_root, split)
    brat_path = os.path.join(data_root, split, "brat")

    # If no standoff annotations exist, make it
    if not os.path.exists(brat_path):
        chemprot_2_standoff(data_dir, brat_path)

    logger.info(f"Loading ChemProt Standoff format from {brat_path}")

    return _BRAT_PARSER.parse(brat_path)


class ChemProtPrompts(DatasetPrompts):
    """
    Create prompts with the ChemProt corpus (https://biocreative.bioinformatics.udel.edu/resources/corpora/chemprot-corpus-biocreative-vi/).
    """

    def __init__(self, data_root: str):
        """Instantiate ChemProt Prompts.

        :param data_root: Root folder containing target dataset
        """
        self.data_root = Path(data_root).resolve()
        self.path = self.data_root / _URL.split("/")[-1]
        self._init_dataset()
        self._init_prompts()

    def _init_dataset(self):
        """
        Download (if unavailable) and prepare the ChemProt dataset

        """
        # confirm dataset dir exist
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
            logger.info(f"Directory {self.data_root} successfully created")

        # download dataset file
        if not os.path.exists(self.path):
            download(_URL, self.path)
            logger.info("Dataset successfully downloaded")
            logger.info(self.path)

        # uncompress file
        full_path = self.data_root / "ChemProt_Corpus"
        if not os.path.exists(full_path):
            uncompress(self.path, self.data_root)

        # Specific to ChemProt, unpack each data split
        data_splits = {"train": _TRAIN, "dev": _DEV, "test": _TEST, "sample": _SAMPLE}

        for dtype in data_splits.values():
            dtype_path = self.data_root / "ChemProt_Corpus" / dtype
            if not os.path.exists(dtype_path):
                logger.info("Unpacking " + str(dtype))
                uncompress(str(dtype_path) + ".zip", full_path)
            else:
                logger.info(str(dtype) + " data is already unzipped.")

        # Create train/dev/test and sample data
        self.splits = {
            split: load_chemprot(full_path, fname)
            for split, fname in data_splits.items()
        }

    def _init_prompts(self):
        """Initialize prompts."""
        self._prompts = {}
        self._metadata = {}

    def _get_pmid(self):
        """
        To analyze ChemProt, the PMIDs are saved in the "BRAT" folder.
        This is useful to keep track of which text files match the inputs, as this may differ from the row order of the original dataset.

        :returns: Dictionary with PMID in order of the data
        """
        return {key: [i.id for i in value] for key, value in self.splits.items()}


def main(args):

    # Make output directory if it doesn't exist
    outpath = Path(args.outdir)
    os.makedirs(outpath, exist_ok=True)

    # Create dataset
    dataset = ChemProtPrompts(args.datadir)

    # ------------------
    # Add NER prompts
    # ------------------
    # ChemProt entities include
    # CHEMICAL - Chemical entity mention type
    # GENE-Y - Gene/Protein mention type with a bio database identifier
    # GENE-N - Gene/Protein mention type w/o bio database identifier
    prompts = {
        "list_chemicals": partial(
            list_comma_separated_entity_mentions, entity_type="CHEMICAL"
        ),
        "list_gene_y": partial(
            list_comma_separated_entity_mentions, entity_type="GENE-Y"
        ),
        "list_gene_n": partial(
            list_comma_separated_entity_mentions, entity_type="GENE-N"
        ),
        "bulleted_list_chemicals": partial(
            bulleted_list_entity_mentions, entity_type="CHEMICAL"
        ),
        "bulleted_list_gene_y": partial(
            bulleted_list_entity_mentions, entity_type="GENE-Y"
        ),
        "bulleted_list_gene_n": partial(
            bulleted_list_entity_mentions, entity_type="GENE-N"
        ),
    }

    for name in prompts:
        dataset.add_prompt(
            prompts[name],
            name,
            answer_keys=None,
            original_task=True,
            answers_in_prompt=True,
            metrics=["f1", "accuracy"],
        )

    # ------------------
    # Add Relational prompts
    # ------------------
    # ChemProt relationships that were group annotated (CPR-3-6, CPR:9)
    # CPR-3: Activator (See above for full annotation)
    # CPR-4: Inhibitor
    # CPR-5: Agonist
    # CPR-6: Antagonist
    # CPR-9: Substrate

    prompts = {
        "list_relations_cpr3": partial(
            list_comma_separated_relations, relation_type="CPR:3"
        ),
        "list_relations_cpr4": partial(
            list_comma_separated_relations, relation_type="CPR:4"
        ),
        "list_relations_cpr5": partial(
            list_comma_separated_relations, relation_type="CPR:5"
        ),
        "list_relations_cpr6": partial(
            list_comma_separated_relations, relation_type="CPR:6"
        ),
        "list_relations_cpr9": partial(
            list_comma_separated_relations, relation_type="CPR:9"
        ),
        "bulleted_list_relations_cpr3": partial(
            bulleted_list_relations, relation_type="CPR:3"
        ),
        "bulleted_list_relations_cpr4": partial(
            bulleted_list_relations, relation_type="CPR:4"
        ),
        "bulleted_list_relations_cpr5": partial(
            bulleted_list_relations, relation_type="CPR:5"
        ),
        "bulleted_list_relations_cpr6": partial(
            bulleted_list_relations, relation_type="CPR:6"
        ),
        "bulleted_list_relations_cpr9": partial(
            bulleted_list_relations, relation_type="CPR:9"
        ),
    }
    for name in prompts:
        dataset.add_prompt(
            prompts[name],
            name,
            answer_keys=None,
            original_task=True,
            answers_in_prompt=True,
            metrics=["accuracy"],
        )

    df = dataset.get_prompts()

    # Save the dataset
    df.to_csv(outpath / "chemprot_prompts.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--datadir", type=str, default="datasets")
    args = parser.parse_args()
    main(args)
