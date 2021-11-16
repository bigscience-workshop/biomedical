import argparse
import json
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
from loguru import logger

from prompts import DatasetPrompts
from utils import download

_HOMEPAGE = "https://github.com/Markus-Zlabinger/pico-annotation"
_DATA_PATH = (
    "https://raw.githubusercontent.com/Markus-Zlabinger/pico-annotation/master/data"
)


class PICOExtractionPrompts(DatasetPrompts):
    """PICOExtractionPrompts class.
    This dataset contains only train data."""

    def __init__(self, data_root):
        """Instantiate PICOExtractionPrompts.
        :param data_root: Root folder containing target dataset
        """
        self.sentence_file = "sentences.json"
        self.annotation_files = {
            "intervention": "annotations/interventions_expert.json",
            "outcome": "annotations/outcomes_expert.json",
            "participant": "annotations/participants_expert.json",
        }
        self.data_root = Path(data_root).resolve()
        self._init_dataset()
        self._init_prompts()
        self.train = {}

    def _init_dataset(self):
        """Download dataset and create splits."""
        # confirm dataset dir exist
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
            logger.info(f"Directory {self.data_root} successfully created")

        # download dataset files
        url = f"{_DATA_PATH}/{self.sentence_file}"
        _path = self.data_root / self.sentence_file
        if not os.path.exists(_path):
            download(url, _path)
            logger.info(f"{self.sentence_file} successfully downloaded")

        for _file in self.annotation_files.values():
            url = f"{_DATA_PATH}/{_file}"
            _path = self.data_root / _file.split("/")[-1]
            if not os.path.exists(_path):
                download(url, _path)
                logger.info(f"{_file} successfully downloaded")

        # load sentences
        _path = self.data_root / self.sentence_file
        with open(_path) as fp:
            sentences = json.load(fp)
            logger.info("Sentences successfully loaded")

        # load annotations
        annotation_dict = {}
        for annotation_type, _file in self.annotation_files.items():
            _path = self.data_root / _file.split("/")[-1]
            with open(_path) as fp:
                annotations = json.load(fp)
                annotation_dict[annotation_type] = annotations
        logger.info("Annotations successfully loaded")

        train = []
        for sentence_id, sentence in sentences.items():
            textual_annotations = {}
            for annotation_type, annotations in annotation_dict.items():
                indices = np.where(
                    np.round(np.mean(annotations[sentence_id]["annotations"], axis=0))
                    == 1
                )[0]
                if len(indices) == 0:
                    annotation_text = "None"
                else:
                    annotation_text = " ".join(
                        [sentence.split()[ind] for ind in indices]
                    )

                textual_annotations[annotation_type] = annotation_text
            sentence_dict = {
                "sentence_id": sentence_id,
                "sentence": sentence,
            }
            sentence_dict.update(textual_annotations)
            train.append(sentence_dict)

        self.splits = {"train": train}

    def _init_prompts(self):
        """Initialize prompts."""
        self._prompts = {}
        self._metadata = {}


def extract_pico(x, pico_type: str):
    """Extraction prompt, x followed by the task.
    Args:
        x: a sentence with annotations from the PICO corpus.
        pico_type: one of the (intervention, outcome, participant).
    Returns:
        the prompt.
    """
    tmpl = "{sentence}\nExtract all {pico_type} spans from the sentence above.\n"
    tmpl += "If there are no {pico_type} mentions, print None.\n|||{target}"
    prompts = {}
    p = tmpl.format(sentence=x["sentence"], pico_type=pico_type, target=x[pico_type])
    prompts[p.lower()] = p

    return list(prompts.values())


def list_pico_elements(x, pico_type: str):
    """Extraction prompt, task followed by x.
    Args:
        x: a sentence with annotations from the PICO corpus.
        pico_type: one of the (intervention, outcome, participant).
    Returns:
        the prompt.
    """
    tmpl = (
        "Create a list of all {pico_type} tokens mentioned in the following sentence. "
    )
    tmpl += (
        'If there are no {pico_type} mentions, print None.\n"{sentence}"\n|||{target}'
    )

    return tmpl.format(pico_type=pico_type, sentence=x["sentence"], target=x[pico_type])


def classify_pico_type(x, pico_type: str):
    """Classification prompt, x, followed by a pico type question.
    Args:
        x: a sentence with annotations from the PICO corpus.
        pico_type: one of the (intervention, outcome, participant).
    Returns:
        the prompt.
    """
    if x[pico_type] != "None":
        tmpl = "{sentence}\n"
        tmpl += 'In the sentence above, is "{extracted_span}" an "intervention", "outcome" or "participant"?'
        tmpl += "\n|||{pico_type}"

        return tmpl.format(
            pico_type=pico_type, sentence=x["sentence"], extracted_span=x[pico_type]
        )


def is_correct_pico_type(x, pico_type: str):
    """Classification prompt, x, followed by a pico type hypothesis.
    Args:
        x: a sentence with annotations from the PICO corpus.
        pico_type: one of the (intervention, outcome, participant).
    Returns:
        the prompt.
    """
    if x[pico_type] != "None":
        tmpl = "{sentence}\n"
        tmpl += 'In the sentence above, is "{extracted_span}" an {pico_type_hypothesis} Yes or No?'
        tmpl += "\n|||{answer}"

        pico_type_hypothesis = random.choice(["intervention", "outcome", "participant"])
        if pico_type_hypothesis == pico_type:
            answer = "Yes"
        else:
            answer = "No"

        return tmpl.format(
            pico_type_hypothesis=pico_type_hypothesis,
            sentence=x["sentence"],
            extracted_span=x[pico_type],
            answer=answer,
        )


def main(args):
    outpath = Path(args.outdir)
    os.makedirs(outpath, exist_ok=True)
    dataset = PICOExtractionPrompts("/tmp/pico-extraction-corpus/")

    prompts = {
        "extract_intervention": partial(extract_pico, pico_type="intervention"),
        "extract_outcome": partial(extract_pico, pico_type="outcome"),
        "extract_participant": partial(extract_pico, pico_type="participant"),
        "list_intervention": partial(list_pico_elements, pico_type="intervention"),
        "list_outcome": partial(list_pico_elements, pico_type="outcome"),
        "list_participant": partial(list_pico_elements, pico_type="participant"),
        "classify_intervention": partial(classify_pico_type, pico_type="intervention"),
        "classify_outcome": partial(classify_pico_type, pico_type="outcome"),
        "classify_participant": partial(classify_pico_type, pico_type="participant"),
        "is_correct_intervention": partial(
            is_correct_pico_type, pico_type="intervention"
        ),
        "is_correct_outcome": partial(is_correct_pico_type, pico_type="outcome"),
        "is_correct_participant": partial(
            is_correct_pico_type, pico_type="participant"
        ),
    }
    for name in prompts:
        dataset.add_prompt(
            prompts[name],
            name,
            answer_keys=None,
            original_task=True,
            answers_in_prompt=False,
            metrics=["f1", "accuracy"],
        )

    df = dataset.get_prompts()
    df = df.dropna(subset=["prompted_x"])
    df.to_csv(f"{outpath}/pico_extraction.tsv", sep="\t", index=False)
    logger.info(f"Prompts created and saved to {outpath}/pico_extraction.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    main(args)
