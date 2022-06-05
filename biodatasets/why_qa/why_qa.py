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

"""
A dataset loader for the n2c2 community-annotated Why Questions dataset.

https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

The dataset consists of a single archive (no splits) and it is available
as a JSON file and as an XLSX file:

    - relations_whyqa_ann-v7-share.json (in SQUAD 2.0 format)
    - relations_whyqa_ann-v7-share.xlsx

The dataset also includes TXT files with the full texts of the
clinical notes.

The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_dataset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files).

Registration AND submission of DUA is required to access the dataset.

[bigbio_schema_name] = qa
"""

import os
import zipfile
import json
from collections import defaultdict
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{,
  author    = {Annotating and Characterizing Clinical Sentences with Explicit Why-{QA} Cues},
  title     = {Fan, Jungwei},
  booktitle = {Proceedings of the 2nd Clinical Natural Language Processing Workshop},
  month     = {jun},
  year      = {2019},
  address   = {Minneapolis, Minnesota, USA},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/W19-1913},
  doi       = {10.18653/v1/W19-1913}

}

}
"""

_DATASETNAME = "[why_qa]"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\

This dataset is a collection of why-questions and their answers generated
from a corpus of clincal notes. The corpus is the 2010 i2b2/VA NLP
challenge and consists of 426 discharge summaries from Partners
Healthcare and Beth Israel Deaconess Medical Center.

"""
_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "External Data User Agreement"

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

def read_zip_file(file_path):
    with zipfile.ZipFile(file_path) as zf:
        with zf.open("n2c2-community-annotations_2010-fan-why-QA/relations_whyqa_ann-v7-share.json") as f:
            dataset = json.load(f)
            return dataset

def _get_samples(dataset):
    samples = dataset['data'][0]['paragraphs']
    return samples

# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
#  Append "Dataset" to the class name: BioASQ --> BioasqDataset
class WhyQaDataset(datasets.GeneratorBasedBuilder):
    """n2c2 community-annotated Why Questions dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)


    BUILDER_CONFIGS = [
        BigBioConfig(
            name="why_qa_source",
            version=SOURCE_VERSION,
            description="why_qa source schema",
            schema="source",
            subset_id="why_qa",
        ),
        BigBioConfig(
            name="why_qa_bigbio_qa",
            version=BIGBIO_VERSION,
            description="why_wa BigBio schema",
            schema="bigbio_qa",
            subset_id="why_qa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "why_qa_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(

            {
            "note_id": datasets.Value("string"),
            "qas": [
                {"question_template": datasets.Value("string"),
                "question": datasets.Value("string"),
                "id": datasets.Value("string"),
                "answers": [
                    {"text": datasets.Value("string"),
                    "answer_start": datasets.Value("int32"),
                    },
                    ],
                "is_impossible": datasets.Value("bool"),
                },
                ],
            "context": datasets.Value("string"),
            },
            )

        elif self.config.schema == "bigbio_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, data_dir, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        dataset = read_zip_file(data_dir)
        samples = _get_samples(dataset)

        if self.config.schema == "source":
            _id = 0
            for sample in samples:
                yield _id, sample
                _id += 1

        elif self.config.schema == "bigbio_[bigbio_schema_name]":
            _id = 0
            for sample in samples:
                for qa in sample['qas']:
                    ans_list = []
                    for answer in qa["answer"]:
                        ans = answer["text"]
                        ans_list.append(ans)
                    bigbio_sample = {
                                        "id" : qa["note_id"],
                                        "question_id" : qa["id"],
                                        "document_id" : sample["note_id"],
                                        "question" : qa["question"],
                                        "type" : qa["question_template"],
                                        "choices" : [],
                                        "context" : sample["context"],
                                        "answer" : ans_list,
                                    }
                    yield _id, bigbio_sample
                    _id += 1


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
