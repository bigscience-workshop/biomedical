# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Tuple, Dict
import pandas as pd
from pathlib import Path
import re

import datasets
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks
from .bigbiohub import kb_features

_LOCAL = True
# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""
_DATASETNAME = "cardiode"

# TODO: Add description of the dataset here
_DESCRIPTION = """\
This dataset is designed for XXX NLP task.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here (if possible)
_LICENSE = "DUA"
_LANGUAGES = "German"
_URLS = {}
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"
_DATASETNAME = "cardiode"
_DISPLAYNAME = "CARDIO:DE"

# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
#  Append "Dataset" to the class name: BioASQ --> BioasqDataset
class CardioDataset(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="cardiode_source",
            version=SOURCE_VERSION,
            description="Cardio:DE source schema",
            schema="source",
            subset_id="cardiode",
        ),
        BigBioConfig(
            name="cardiode_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Cardio:DE BigBio schema",
            schema="bigbio_kb",
            subset_id="cardidoe",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cardiode_bigbio_kb"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            # TODO: Create your source schema here
            raise NotImplementedError()

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir),
                    "split": "train",
                },
            )
        ]


    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            raise NotImplementedError()
        elif self.config.schema == "bigbio_kb":
            doc_ids = _sort_files(Path(filepath) / 'tsv' / 'CARDIODE400_main')
            for uid, doc in enumerate(doc_ids):
                tsv_path = Path(filepath) / 'tsv' / 'CARDIODE400_main' / f'{doc}'
                txt_path = Path(filepath) / 'txt' / 'CARDIODE400_main' / f'{doc.replace(".tsv", ".txt")}'
                df = _parse_tsv(tsv_path)
                original, no_jumps = _parse_txt(txt_path)
                yield uid, _make_bigbio_kb(uid, doc, df, original, no_jumps)


def _parse_tsv(path: str) -> pd.DataFrame:
    # read whole .tsv as a string
    with open(path, encoding='utf-8') as file:
        content = file.read()

    # separate doc into sentences
    passages = content.split('#')

    # remove the first line (un-tabbed) of each sentence
    # and split sentences into words/tokens
    for i, passage in enumerate(passages):
        passages[i] = passage.split('\n')[1:]

    # clean empty sentences and tokens
    clean_passages = [[token for token in passage if token != ''] for passage in passages if passage != []]

    # make a dataframe out of the clean tokens
    df = []
    for passage in clean_passages:
        for token in passage:
            df.append(token.split('\t'))

    df = pd.DataFrame(df).rename(columns={
        0: 'passage_token_id',
        1: 'token_offset',
        2: 'text',
        3: 'label',
        4: 'idk',
        5: 'relation',
        6: 'section',
    })

    # delete weird rows were label is NoneType
    df = df[df['label'].notnull()]

    # split passage and token ids
    df[['passage_id', 'token_id']] = df['passage_token_id'].str.split('-', expand=True)

    # split labels and their spans
    # some docs do not have labels spanning various tokens (or they do not have any labels at all)
    if df['label'].apply(lambda x: '[' in x).any():
        df[['lab', 'span']] = df['label'].str.split('[', expand=True)
        df['span'] = df['span'].str.replace(']', "", regex=True)
    else:
        df['lab'] = '_'
        df['span'] = None

    return df.drop(columns=['label', 'idk', 'token_id'])


def _parse_txt(path: str) -> (str, str):
    with open(path, encoding='utf-8') as file:
        original = file.read()
        # remove consecutive \n
        no_jumps = re.sub("\n{2,}", "\n", original)
        return original, no_jumps


def _make_bigbio_kb(uid: int, doc_id: str, df: pd.DataFrame, original: str, no_jumps: str):
    out = {
        'id': str(uid),
        'document_id': doc_id,
        'passages': [],
        'entities': [],
        'events': [],
        'coreferences': [],
        'relations': [],
    }

    # handle passages
    i = 0
    sen_num = 0
    while i < len(df):
        pid = df.iloc[i]['passage_id']
        passage = df[df['passage_id'] == pid]

        out['passages'].append({
            'id': f'{uid}-{pid}',
            'type': 'sentence',
            'text': [no_jumps.split('\n')[int(pid) - 1]],
            'offsets': [[int(passage.iloc[0]['token_offset'].split('-')[0]),
                         int(passage.iloc[-1]['token_offset'].split('-')[1])]],
        })

        i += len(passage)
        sen_num += 1

    # handle entities
    i = 0
    while i < len(df):
        if df.iloc[i]['lab'] != "_" and df.iloc[i]['span'] is None:
            out['entities'].append({
                "id": f'{uid}-{df.iloc[i]["passage_token_id"]}',
                "type": df.iloc[i]["lab"],
                "text": [original[int(df.iloc[i]['token_offset'].split('-')[0]):int(df.iloc[i]['token_offset'].split('-')[1])]],
                "offsets": [[int(x) for x in df.iloc[i]['token_offset'].split('-')]],
                "normalized": [],
            })
            i += 1
        elif df.iloc[i]['span'] is not None:
            ent = df[df['span'] == df.iloc[i]['span']]
            out['entities'].append({
                "id": f'{uid}-{df.iloc[i]["passage_token_id"]}',
                "type": df.iloc[i]["lab"],
                "text": [original[int(ent.iloc[0]['token_offset'].split('-')[0]):int(ent.iloc[-1]['token_offset'].split('-')[1])]],
                "offsets": [[int(ent.iloc[0]['token_offset'].split('-')[0]),
                             int(ent.iloc[-1]['token_offset'].split('-')[1])]],
                "normalized": [],
            })
            i += len(ent)
        else:
            i += 1

    return out


def _sort_files(filepath):
    doc_ids = os.listdir(filepath)
    doc_ids = [int(doc_ids[i].split('.')[0]) for i in range(len(doc_ids))]
    doc_ids = sorted(doc_ids)
    doc_ids = [f'{doc_ids[i]}.tsv' for i in range(len(doc_ids))]
    return doc_ids


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
