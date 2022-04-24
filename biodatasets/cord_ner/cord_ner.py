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
CORD-NER dataset covers 75 fine-grained entity types: In addition to the common biomedical entity types (e.g., genes, chemicals and diseases), it covers many new entity types related explicitly to the COVID-19 studies (e.g., coronaviruses, viral proteins, evolution, materials, substrates and immune responses), which may benefit research on COVID-19 related virus, spreading mechanisms, and potential vaccines. CORD-NER annotation is a combination of four sources with different NER methods.
"""

import os
import json
from typing import List, Tuple, Dict

import datasets
from biomed_datasets.utils import schemas
from biomed_datasets.utils.configs import BigBioConfig
from biomed_datasets.utils.constants import Tasks

_CITATION = """\
@article{DBLP:journals/corr/abs-2003-12218,
  author    = {Xuan Wang and
               Xiangchen Song and
               Yingjun Guan and
               Bangzheng Li and
               Jiawei Han},
  title     = {Comprehensive Named Entity Recognition on {CORD-19} with Distant or
               Weak Supervision},
  journal   = {CoRR},
  volume    = {abs/2003.12218},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.12218},
  eprinttype = {arXiv},
  eprint    = {2003.12218},
  timestamp = {Fri, 08 May 2020 13:20:46 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2003-12218.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


_DATASETNAME = "cord_ner"


_DESCRIPTION = """\
CORD-NER dataset covers 75 fine-grained entity types: In addition to the common biomedical entity types (e.g., genes, chemicals and diseases), it covers many new entity types related explicitly to the COVID-19 studies (e.g., coronaviruses, viral proteins, evolution, materials, substrates and immune responses), which may benefit research on COVID-19 related virus, spreading mechanisms, and potential vaccines. CORD-NER annotation is a combination of four sources with different NER methods.
"""

_HOMEPAGE = "https://xuanwang91.github.io/2020-03-20-cord19-ner/"


_LICENSE = """
    This dataset is made from multiple datasets by Allen Institute 
    for AI in partnership with the Chan Zuckerberg Initiative, Georgetown 
    University’s Center for Security and Emerging Technology, Microsoft Research, 
    IBM, and the National Library of Medicine - National Institutes 
    of Health, in coordination with The White House Office of Science 
    and Technology Policy . The licenses are different depending on the source.
    The full license details can be found here: https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv
    """

_URLS = {
    _DATASETNAME: {
        "full": "https://uofi.app.box.com/index.php?rm=box_download_shared_file&shared_name=k8pw7d5kozzpoum2jwfaqdaey1oij93x&file_id=f_651148518303",
        "ner": "https://uofi.app.box.com/index.php?rm=box_download_shared_file&shared_name=k8pw7d5kozzpoum2jwfaqdaey1oij93x&file_id=f_642495001609",
        "corpus": "https://uofi.app.box.com/index.php?rm=box_download_shared_file&shared_name=k8pw7d5kozzpoum2jwfaqdaey1oij93x&file_id=f_642522056185",
    }
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class CordNERDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="cord_ner_ner_source",
            version=SOURCE_VERSION,
            description="cord_ner source schema for ner file",
            schema="source",
            subset_id="cord_ner",
        ),
        BigBioConfig(
            name="cord_ner_corpus_source",
            version=SOURCE_VERSION,
            description="cord_ner source schema for corpus file",
            schema="source",
            subset_id="cord_ner",
        ),
        BigBioConfig(
            name="cord_ner_source",
            version=SOURCE_VERSION,
            description="cord_ner source schema for full file",
            schema="source",
            subset_id="cord_ner",
        ),
        BigBioConfig(
            name="cord_ner_bigbio_kb",
            version=BIGBIO_VERSION,
            description="cord_ner BigBio schema",
            schema="bigbio_kb",
            subset_id="cord_ner",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cord_ner_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            if self.config.name == "cord_ner_source":
                features = datasets.Features(
                    {
                        "id": datasets.Value("int32"),
                        "source": datasets.Value("string"),
                        "doi": datasets.Value("string"),
                        "pmcid": datasets.Value("string"),
                        "pubmed_id": datasets.Value("string"),
                        "publish_time": datasets.Value("string"),
                        "authors": datasets.Value("string"),
                        "journal": datasets.Value("string"),
                        "title": datasets.Value("string"),
                        "abstract": datasets.Value("string"),
                        "body": datasets.Value("string"),
                        "entities": [
                            {
                                "end": datasets.Value("int32"),
                                "start": datasets.Value("int32"),
                                "text": datasets.Value("string"),
                                "type": datasets.Value("string"),
                            }
                        ],
                    }
                )
            elif self.config.name == "cord_ner_ner_source":
                features = datasets.Features(
                    {
                        "doc_id": datasets.Value("int32"),
                        "sents": [
                            {
                                "entities": [
                                    {
                                        "end": datasets.Value("int32"),
                                        "start": datasets.Value("int32"),
                                        "text": datasets.Value("string"),
                                        "type": datasets.Value("string"),
                                    }
                                ],
                                "sent_id": datasets.Value("int32"),
                            }
                        ],
                    }
                )
            elif self.config.name == "cord_ner_corpus_source":
                features = datasets.Features(
                    {
                        "doc_id": datasets.Value("int32"),
                        "sents": [
                            {
                                "sent_id": datasets.Value("int32"),
                                "sent_tokens": datasets.Sequence(
                                    datasets.Value("string")
                                ),
                            }
                        ],
                        "source": datasets.Value("string"),
                        "doi": datasets.Value("string"),
                        "pmcid": datasets.Value("string"),
                        "pubmed_id": datasets.Value("string"),
                        "publish_time": datasets.Value("string"),
                        "authors": datasets.Value("string"),
                        "journal": datasets.Value("string"),
                    }
                )
        elif self.config.name == "cord_ner_bigbio_kb":
            features = schemas.kb_features
        else:
            raise NotImplementedError(f"{self.config.name} not a valid config name")

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
            # The download method may not be reliable, so if it fails this function will still work for local files
            try:

                if (
                    self.config.name == "cord_ner_source"
                    or self.config.name == "cord_ner_bigbio_kb"
                ):
                    url = {"train": _URLS[_DATASETNAME]["full"]}
                elif self.config.name == "cord_ner_ner_source":
                    url = {"train": _URLS[_DATASETNAME]["ner"]}
                elif self.config.name == "cord_ner_corpus_source":
                    url = {"train": _URLS[_DATASETNAME]["corpus"]}
                data_dir = dl_manager.download_and_extract(url)
            except:
                raise ConnectionError(
                    "The dataset could not be downloaded. Please download to local storage and pass the data_dir kwarg to load_dataset."
                )
        else:
            filenames = {
                "full": "CORD-NER-full.json",
                "ner": "CORD-NER-ner.json",
                "corpus": "CORD-NER-corpus.json",
            }

            if (
                self.config.name == "cord_ner_source"
                or self.config.name == "cord_ner_bigbio_kb"
            ):
                data_dir = {
                    "train": os.path.join(self.config.data_dir, filenames["full"])
                }
            elif self.config.name == "cord_ner_ner_source":
                data_dir = {
                    "train": os.path.join(self.config.data_dir, filenames["ner"])
                }
            elif self.config.name == "cord_ner_corpus_source":
                data_dir = {
                    "train": os.path.join(self.config.data_dir, filenames["corpus"])
                }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath) as fp:
                for key, example in enumerate(fp.readlines()):
                    yield key, json.loads(example)

        elif self.config.schema == "bigbio_kb":

            with open(filepath) as fp:
                unq_id = 0
                for key, line in enumerate(fp.readlines()):

                    example_id = unq_id
                    unq_id += 1

                    line = json.loads(line)
                    passages = []
                    offset_start = 0
                    for text_type in ["title", "abstract", "body"]:

                        passages.append(
                            {
                                "id": str(unq_id),
                                "type": text_type,
                                "text": [line[text_type]],
                                "offsets": [
                                    [
                                        offset_start,
                                        offset_start + len(line[text_type]),
                                    ]
                                ],
                            }
                        )
                        # add +1 for the space
                        offset_start += len(line[text_type]) + 1
                        unq_id += 1

                    entities = []
                    for ent in line["entities"]:
                        entities.append(
                            {
                                "id": str(unq_id),
                                "type": ent["type"],
                                "text": [ent["text"]],
                                "offsets": [[ent["start"], ent["end"]]],
                                "normalized": [
                                    {
                                        "db_name": "",
                                        "db_id": "",
                                    }
                                ],
                            }
                        )
                        unq_id += 1

                    yield key, {
                        "id": str(example_id),
                        "document_id": str(line["id"]),
                        "passages": passages,
                        "entities": entities,
                        "events": [],
                        "coreferences": [],
                        "relations": [],
                    }


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
