import json
import os
import argparse
from pathlib import Path
from loguru import logger
from functools import partial
from utils import download
from prompts import DatasetPrompts
import numpy as np

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
            "interventions": "annotations/interventions_expert.json",
            "outcomes": "annotations/outcomes_expert.json",
            "participants": "annotations/participants_expert.json",
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
        pico_type: one of the (interventions, outcomes, participants).
    Returns:
        the prompt.
    """
    tmpl = "{sentence}\nExtract all {pico_type} spans from the sentence above.\n|||{target}"
    prompts = {}
    p = tmpl.format(sentence=x["sentence"], pico_type=pico_type, target=x[pico_type])
    prompts[p.lower()] = p

    return list(prompts.values())


def list_pico_elements(x, pico_type: str):
    """Extraction prompt, task followed by x.
    Args:
        x: a sentence with annotations from the PICO corpus.
        pico_type: one of the (interventions, outcomes, participants).
    Returns:
        the prompt.
    """
    tmpl = "Create a space-separated list of all {pico_type} tokens mentioned in the following sentence. "
    tmpl += (
        'If there are no {pico_type} mentions, print None.\n"{sentence}"\n|||{target}'
    )

    return tmpl.format(pico_type=pico_type, sentence=x["sentence"], target=x[pico_type])


def main(args):
    outpath = Path(args.outdir)
    os.makedirs(outpath, exist_ok=True)
    dataset = PICOExtractionPrompts("/tmp/pico-extraction-corpus/")

    prompts = {
        "extract_interventions": partial(extract_pico, pico_type="interventions"),
        "extract_outcomes": partial(extract_pico, pico_type="outcomes"),
        "extract_participants": partial(extract_pico, pico_type="participants"),
        "list_interventions": partial(list_pico_elements, pico_type="interventions"),
        "list_outcomes": partial(list_pico_elements, pico_type="outcomes"),
        "list_participants": partial(list_pico_elements, pico_type="participants"),
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

    df = dataset.get_prompts()
    df.to_csv(f"{outpath}/pico_extraction.tsv", sep="\t", index=False)
    logger.info(f"Prompts created and saved to {outpath}/pico_extraction.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    main(args)
