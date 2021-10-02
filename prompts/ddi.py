"""
DDI Corpus

María Herrero-Zazo, Isabel Segura-Bedmar, Paloma Martínez, Thierry Declerck, The DDI corpus:
An annotated corpus with pharmacological substances and drug–drug interactions, Journal of Biomedical Informatics,
Volume 46, Issue 5, October 2013, Pages 914-920, http://dx.doi.org/10.1016/j.jbi.2013.07.011.

Usage:
python ddi.py --outdir biomedical/

"""
import os
import argparse
from pathlib import Path
from loguru import logger
from functools import partial
from typing import List, Callable
from pybrat.parser import BratParser, Example
from utils import download, uncompress
from prompts import DatasetPrompts

# TODO: Implement proper Hugging Face's Datasets version of DDICorpus
_HOMEPAGE = "https://github.com/isegura/DDICorpus"
_URL = "https://github.com/isegura/DDICorpus/raw/master/DDICorpus-2013(BRAT).zip"
_TRAINING_PATH = "Train"
_TEST_PATH = "Test"
_BRAT_PARSER = BratParser(error="ignore")


def load_ddi_corpus_split(path: str) -> List[Example]:
    logger.info(f"Loading DDICorpus split from {path}")
    return _BRAT_PARSER.parse(path)


class DDICorpusPrompts(DatasetPrompts):
    """Prompts creator from DDI Corpus."""

    def __init__(self, data_root: str):
        """Instantiate DDICorpusPrompts.

        Args:
            data_root: dataset download root folder.
        """
        self.data_root = Path(data_root).resolve()
        self.path = self.data_root / _URL.split("/")[-1]
        self._init_dataset()
        self._init_prompts()

    def _init_dataset(self):
        """Download dataset and create splits."""
        # confirm dataset dir exist
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
            logger.info(f"Directory {self.data_root} successfully created")

        # download dataset file
        if not os.path.exists(self.path):
            download(_URL, self.path)
            logger.info("Dataset successfully downloaded")

        # uncompress file
        full_path = self.data_root / "DDICorpusBrat"
        if not os.path.exists(full_path):
            uncompress(self.path, self.data_root)

        self.splits = {
            split: load_ddi_corpus_split(str(full_path / path))
            for split, path in {"train": _TRAINING_PATH, "test": _TEST_PATH}.items()
        }

    def _init_prompts(self):
        """Initialize prompts."""
        self._prompts = {}
        self._metadata = {}


# NOTE: implemented prompts
def list_entity_template(
    example: Example,
    entity_type: str,
    join_string: str = ", ",
    mention_fn: Callable[str, str] = lambda mention: mention,
) -> str:
    """Entity listing prompt generation.

    Args:
        example: an example from BRAT.
        entity_type: entity type.
        join_string: string used for joining. Defaults to ", ".
        mention_fn: function to process the mention. Defaults to identity.

    Returns:
        the prompt.
    """
    template = "List of all {entity_type} names mentioned in the following text. "
    template += 'If there are no {entity_type} mentions, print None.{line_separator}"{text}"{line_separator}|||{target}'
    target = join_string.join(
        [
            mention_fn(entity.mention)
            for entity in example.entities
            if entity.type == entity_type
        ]
    )
    return template.format(
        entity_type=entity_type,
        text=example.text,
        target=target if target else "None",
        line_separator=os.linesep,
    )


def list_comma_separated_entity_mentions(example: Example, entity_type: str) -> str:
    """List comma-separated entity mentions.

    Args:
        example: an example from BRAT.
        entity_type: entity type.

    Returns:
        the prompt.
    """
    return list_entity_template(example=example, entity_type=entity_type)


def bulleted_list_entity_mentions(example: Example, entity_type: str) -> str:
    """Bulleted list entity mentions.

    Args:
        example: an example from BRAT.
        entity_type: entity type.

    Returns:
        the prompt.
    """
    return list_entity_template(
        example=example,
        entity_type=entity_type,
        join_string=" -",
        mention_fn=lambda mention: f"{mention}{os.linesep}",
    )


def main(args):

    outpath = Path(args.outdir)
    os.makedirs(outpath, exist_ok=True)
    dataset = DDICorpusPrompts("/tmp/ddi-corpus/")

    # entity types: {'BRAND', 'DRUG', 'DRUG_N', 'GROUP'}
    prompts = {
        "list_drugs": partial(list_comma_separated_entity_mentions, entity_type="DRUG"),
        "list_brands": partial(
            list_comma_separated_entity_mentions, entity_type="BRAND"
        ),
        "list_drug_ns": partial(
            list_comma_separated_entity_mentions, entity_type="DRUG_N"
        ),
        "list_groups": partial(
            list_comma_separated_entity_mentions, entity_type="GROUP"
        ),
        "bulleted_list_drugs": partial(
            bulleted_list_entity_mentions, entity_type="DRUG"
        ),
        "bulleted_list_brands": partial(
            bulleted_list_entity_mentions, entity_type="BRAND"
        ),
        "bulleted_list_drug_ns": partial(
            bulleted_list_entity_mentions, entity_type="DRUG_N"
        ),
        "bulleted_list_groups": partial(
            bulleted_list_entity_mentions, entity_type="GROUP"
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

    # TODO: # relation types: {'ADVISE', 'EFFECT', 'INT', 'MECHANISM'}
    df = dataset.get_prompts()
    df.to_csv(outpath / "ddi_corpus_prompts.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    main(args)
