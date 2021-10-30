"""
n2c2 Smoking Status Identification Challenge
Özlem Uzuner, PhD, Ira Goldstein, MBA, Yuan Luo, MS, Isaac Kohane, MD, PhD, Identifying Patient Smoking Status from Medical Discharge Records, Journal of the American Medical Informatics Association, Volume 15, Issue 1, January 2008, Pages 14–24, https://doi.org/10.1197/jamia.M2408
Usage:
python n2c2_smoking.py --indir data_dir/ --outdir output_dir/
"""
import os
import argparse
from pathlib import Path
from loguru import logger
from functools import partial
import xml.etree.ElementTree as et
from prompts import DatasetPrompts

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"
_TRAINING_FILE = "smokers_surrogate_train_all_version2.xml"
_TEST_FILE = "smokers_surrogate_test_all_groundtruth_version2.xml"

def load_smoking_corpus(path):
    """
    Load the data split.
    :param path: train/test data path
    """
    logger.info(f"Loading n2c2 Smoking Status data from {path}")
    with open(path) as raw:
        file = raw.read().strip()
    root = et.fromstring(file)
    ids = []
    notes = []
    labels = []
    documents = root.findall("./RECORD")
    for document in documents:
        ids.append(document.attrib['ID'])
        notes.append(document.findall('./TEXT')[0].text)
        labels.append(document.findall('./SMOKING')[0].attrib['STATUS'].lower())

    return [(id,note,label) for id, note, label in zip(ids,notes,labels)]
    

class SmokingCorpusPrompts(DatasetPrompts):
    """
    Create prompts from n2c2 smoking status corpus
    """
    def __init__(self, data_root):
        """Instantiate SmokingCorpusPrompts.
        :param data_root: Root folder containing target dataset
        """
        self.data_root = Path(data_root).resolve()
        self._init_dataset()
        self._init_prompts()

    def _init_dataset(self):
        """
        Prepare smoking status data.
        Download data manually from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/", which requires user registration and DUA.
        """
        self.splits = {
            split:load_smoking_corpus(self.data_root / fname)
            for split,fname in {
                'train':_TRAINING_FILE,
                'test':_TEST_FILE
            }.items()
        }
        logger.info("Dataset successfully loaded")

    def _init_prompts(self):
        """Initialize prompts."""
        self._prompts = {}
        self._metadata = {}


#
# Create Prompts + Helper Functions
#
def classify(x):
    """Simple prompt for classification.
    Args:
        x: a note from the smoking status corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "What smoking status would you give this patient from reading the record?\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[2])

def classify_question_first(x):
    """Simple prompt for classification; question followed by x.
    Args:
        x: a note from the smoking status corpus.
    Returns:
        the prompt.
    """
    tmpl = "What smoking status would you give this patient from reading the record?\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[2])

def classify_with_choices_question_first_v1(x):
    """Classification prompt with answer choices provided (v1); question followed by x.
    Args:
        x: a note from the smoking status corpus.
    Returns:
        the prompt.
    """
    tmpl = "Based on the explicitly stated smoking-related facts in the record, "
    tmpl += "would you label this patient as \"current smoker\", \"non-smoker\", \"past smoker\", \"smoker\" or \"unknown\"? \n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[2])

def classify_with_choices_v1(x):
    """Classification prompt with answer choices provided (v1).
    Args:
        x: a note from the smoking status corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on the explicitly stated smoking-related facts in the record, "
    tmpl += "would you label this patient as \"current smoker\", \"non-smoker\", \"past smoker\", \"smoker\" or \"unknown\"? \n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[2])

def classify_with_choices_question_first_v2(x):
    """Classification prompt with answer choices provided (v2); question followed by x.
    Args:
        x: a note from the smoking status corpus.
    Returns:
        the prompt.
    """
    tmpl = "Given 5 smoking status categories, namely \"current smoker\", \"non-smoker\", \"past smoker\", \"smoker\" and \"unknown\", "
    tmpl += "which one category that best describes this patient?\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[2])

def classify_with_choices_v2(x):
    """Classification prompt with answer choices provided (v2).
    Args:
        x: a note from the smoking status corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl = "Given 5 smoking status categories, namely \"current smoker\", \"non-smoker\", \"past smoker\", \"smoker\" and \"unknown\", "
    tmpl += "which one category that best describes this patient?\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[2])



def main(args):

    outpath = Path(args.outdir)
    dataset = SmokingCorpusPrompts(args.indir)

    # Add prompts
    prompts = {
        "classify_note":partial(classify),
        "classify_question_first":partial(classify_question_first),
        "classify_with_choices_v1":partial(classify_with_choices_v1),
        "classify_with_choices_question_first_v1":partial(classify_with_choices_question_first_v1),
        "classify_with_choices_v2":partial(classify_with_choices_v2),
        "classify_with_choices_question_first_v2":partial(classify_with_choices_question_first_v2),
    }
    for name in prompts:
        if name in ["classify_note", "classify_question_first"]:
            answers_in_prompt_ = False
        else:
            answers_in_prompt_ = True
            
        dataset.add_prompt(prompts[name],
                        name,
                        answer_keys=['current smoker', 'non-smoker', 'past smoker', 'smoker', 'unknown'],
                        original_task=True,
                        answers_in_prompt=answers_in_prompt_,
                        metrics=['f1','accuracy'])

    df = dataset.get_prompts()

    # Save the dataset
    df.to_csv(outpath / 'smoking_prompts.tsv', sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    main(args)
