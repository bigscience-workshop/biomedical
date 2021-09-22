"""BioCreative V Chemical Disease Relation (CDR) Task"""
import os
import argparse
import itertools
import collections
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from utils import (
    load_bioc_corpus,
    download,
    uncompress,
    create_yes_no_choices_from_relations
)
from .prompts import DatasetPrompts

# TODO: Implement proper Hugging Face's Datasets version of BC5CDR
_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/"
_URL = "http://www.biocreative.org/media/store/files/2016/CDR_Data.zip"
_TRAINING_FILE = "CDR_TrainingSet.BioC.xml"
_DEV_FILE = "CDR_DevelopmentSet.BioC.xml"
_TEST_FILE = "CDR_TestSet.BioC.xml"

class Bc5cdrCorpusPrompts(DatasetPrompts):

    def __init__(self, data_root):

        self.data_root = Path(data_root).resolve()
        self.path = self.data_root / _URL.split('/')[-1]
        self._init_dataset()
        self._init_prompts()

    def _init_dataset(self):
        """Download dataset and create splits"""
        # confirm dataset dir exist
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
            print(f"Directory {self.data_root} successfully created")

        # download dataset file
        if not os.path.exists(self.path):
            download(_URL, self.path)
            print(f"Dataset successfully downloaded")

        # uncompress file
        full_path = self.data_root / 'CDR_Data' / 'CDR.Corpus.v010516'
        if not os.path.exists(full_path):
            uncompress(self.path, self.data_root)

        self.splits = {
            split:load_bioc_corpus(full_path / fname, 'MESH')
            for split,fname in {
                'train':_TRAINING_FILE,
                'dev':_DEV_FILE,
                'test':_TEST_FILE
            }.items()
        }

    def _init_prompts(self):
        self._prompts = {}
        self._metadata = {}

    # def add_prompt(self,
    #                func,
    #                name,
    #                answer_keys=None,
    #                original_task=False,
    #                answers_in_prompt=False,
    #                metrics=None):
    #
    #     self._prompts[name] = func
    #     self._metadata[name] = {
    #         'answer_keys': answer_keys,
    #         'original_task': original_task,
    #         'answers_in_prompt': answers_in_prompt,
    #         'metrics': metrics
    #     }
    #
    # def get_prompts(self):
    #     """
    #     Create a pandas dataframe for prompts
    #     :return:
    #     """
    #     data = []
    #     for split,dataset in self.splits.items():
    #         for name in self._prompts:
    #             # create prompted instances
    #             f = self._prompts[name]
    #             prompts = [f(x) for x in dataset]
    #             # multiple prompts per instance
    #             if len([True for x in prompts if type(x) is list]) > 1:
    #                 prompts = list(itertools.chain.from_iterable(prompts))
    #
    #             names = np.array([split, name] * len(prompts)).reshape(-1,2)
    #             prompts = np.array(prompts).reshape(-1,1)
    #             prompts = np.hstack((names, prompts))
    #             print(prompts.shape)
    #             data.append(prompts)
    #
    #             # generate metadata
    #             # TODO: dump to file
    #             # m = self._metadata[name]
    #             # row = [
    #             #     split,
    #             #     name,
    #             #     '|||'.join(m['answer_keys']) if m['answer_keys'] else m['answer_keys'],
    #             #     m['original_task'],
    #             #     m['answers_in_prompt'],
    #             #     '|||'.join(m['metrics']) if m['metrics'] else m['metrics']
    #             # ]
    #
    #     return pd.DataFrame(data=np.vstack(data),
    #                         columns=['split', 'prompt_name', 'prompted_x'])


#
# Create Prompts + Helper Functions
#

def list_entity_mentions(x, entity_type):
    """Assumes BioC document + annotations as input instance x"""
    tmpl = "Create a comma-separated list of all {entity_type} names mentioned in the following PubMed abstract. "
    tmpl += 'If there are no {entity_type} mentions, print None.\n"{text}"\n|||{target}'
    target = ", ".join([e.text for e in x.ents if e.type_ == entity_type])
    return tmpl.format(entity_type=entity_type, text=x.text, target=target if target else "None")

def bulleted_list_entity_mentions(x, entity_type):
    """Assumes BioC document + annotations as input instance x"""
    tmpl = "Create a comma-separated list of all {entity_type} names mentioned in the following PubMed abstract. "
    tmpl += 'If there are no {entity_type} mentions, print None.\n"{text}"\n|||{target}'
    target = " -".join([f"{e.text}\n" for e in x.ents if e.type_ == entity_type])
    return tmpl.format(entity_type=entity_type, text=x.text, target=target if target else "None")

def yes_no_cid(x, rela_types):
    targets = create_yes_no_choices_from_relations(x, rela_types)
    tmpl = 'Read the following PubMed abstract and answer the provided question: \n"{text}"\nCan {disease} be induced by {chemical}? Yes or No?|||{target}'
    prompts = {}
    for rela in rela_types:
        if rela not in targets:
            continue
        for cid in targets[rela]['pos']:
            for chemical,disease in cid:
                p = tmpl.format(text=x.text, disease=disease, chemical=chemical, target='Yes')
                prompts[p.lower()] = p
        for cid in targets[rela]['neg']:
            for chemical,disease in cid:
                p = tmpl.format(text=x.text, disease=disease, chemical=chemical, target='No')
                prompts[p.lower()] = p
    return list(prompts.values())

def yes_no_cid_closed_book(x, rela_types):
    targets = create_yes_no_choices_from_relations(x, rela_types)
    tmpl = 'Can {disease} be induced by {chemical}? Yes or No?|||{target}'
    prompts = {}
    for rela in rela_types:
        if rela not in targets:
            continue
        for cid in targets[rela]['pos']:
            for chemical,disease in cid:
                p = tmpl.format(disease=disease, chemical=chemical, target='Yes')
                prompts[p.lower()] = p
        for cid in targets[rela]['neg']:
            for chemical,disease in cid:
                p = tmpl.format(disease=disease, chemical=chemical, target='No')
                prompts[p.lower()] = p
    return list(prompts.values())


def main(args):

    outpath = Path(args.outdir)
    dataset = Bc5cdrCorpusPrompts("../datasets/")

    # NER as sequence generation where we emit all entities in an abstract
    # TODO: Should this count as the original task?
    prompts = {
        "list_chemicals":partial(list_entity_mentions, entity_type='Chemical'),
        "list_diseases":partial(list_entity_mentions, entity_type='Disease'),
        "bulleted_list_chemicals":partial(bulleted_list_entity_mentions, entity_type='Chemical'),
        "bulleted_list_diseases":partial(bulleted_list_entity_mentions, entity_type='Disease'),
    }
    for name in prompts:
        dataset.add_prompt(prompts[name],
                           name,
                           answer_keys=None,
                           original_task=True,
                           answers_in_prompt=True,
                           metrics=['f1','accuracy'])

    # Transform relation extraction into a Yes/No QA format
    prompts = {
        "yes_no_cid": partial(yes_no_cid, rela_types=[('Chemical','Disease')]),
        "yes_no_cid_closed_book": partial(yes_no_cid_closed_book, rela_types=[('Chemical','Disease')])
    }
    for name in prompts:
        dataset.add_prompt(prompts[name],
                           name,
                           answer_keys=['Yes','No'],
                           original_task=False,
                           answers_in_prompt=True,
                           metrics=['accuracy'])

    df = dataset.get_prompts()
    df.to_csv(outpath / 'bc5cdr_prompts.tsv', sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    main(args)