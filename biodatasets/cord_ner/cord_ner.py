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
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

[bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""

import os
import json
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

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

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: {
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

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    # TODO: For each dataset, implement Config for Source and BigBio;
    #  If dataset contains more than one subset (see examples/bioasq.py) implement for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. bioasq10b_(source|bigbio)_[bigbio_schema_name]
    #   - version: option = (SOURCE_VERSION|BIGBIO_VERSION)
    #   - description: one line description for the dataset
    #   - schema: options = (source|bigbio_[bigbio_schema_name])
    #   - subset_id: subset id is the canonical name for the dataset (eg. bioasq10b)
    #  where [bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="cord_ner_ner_source",
            version=SOURCE_VERSION,
            description="cord_ner source schema for ner file",
            schema="source",
            subset_id="cord_ner_ner",
        ),
        BigBioConfig(
            name="cord_ner_corpus_source",
            version=SOURCE_VERSION,
            description="cord_ner source schema for corpus file",
            schema="source",
            subset_id="cord_ner_corpus",
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
            if self.config.subset_id == "cord_ner_ner":
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
            elif self.config.subset_id == "cord_ner_corpus":
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
                        "publish_time": datasets.Value("string"),
                        "authros": datasets.Value("string"),
                        "journal": datasets.Value("string"),
                    }
                )
            else:
                raise NotImplementedError(
                    f"{self.config.subset_id} not a valid config subset_id"
                )

        elif self.config.schema == "bigbio_kb":
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
            # The download method may not be reliable, so if it fails
            try:
                if self.config.subset_id == "cord_ner_ner":
                    urls = _URLS[_DATASETNAME]["ner"]
                elif self.config.subset_id == "cord_ner_corpus":
                    urls = _URLS[_DATASETNAME]["corpus"]
                elif self.config.name == "cord_ner_bigbio_kb":
                    urls = [url for url in _URLS[_DATASETNAME].values()]

                data_dir = dl_manager.download_and_extract(urls)
            except:
                raise ConnectionError(
                    "The dataset could not be downloaded. Please download to local storage and pass the data_dir kwarg to load_dataset."
                )
        else:
            data_dir = self.config.data_dir

        if (
            self.config.subset_id == "cord_ner_ner"
            or self.config.name == "cord_ner_bigbio_kb"
        ):
            filepath = "CORD-NER-ner.json"
        elif self.config.subset_id == "cord_ner_corpus":
            filepath = "CORD-NER-corpus.json"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, filepath),
                    "split": "train",
                },
            ),
        ]

    def _corpus_to_dict(self):

        doc2sents = {}
        with open(os.path.join(self.config.data_dir, "CORD-NER-corpus.json")) as fp:
            for line in fp.readlines():
                line = json.loads(line)
                doc_id = line["doc_id"]

                if doc_id not in doc2sents:
                    doc2sents[doc_id] = []

                doc2sents[doc_id].append(line["sents"])

        return doc2sents

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, "r") as fp:
                for key, example in enumerate(fp.readlines()):
                    yield key, json.loads(example)

        elif self.config.schema == "bigbio_[bigbio_schema_name]":

            doc2sents = self._corpus_to_dict()
            with open(filepath, "r") as fp:
                for key, line in enumerate(fp.readlines()):
                    line = json.loads(line)
                    corpus_sents = doc2sents[line["doc_id"]]

                    passage_offset_start = 0
                    passages = []
                    for sent in corpus_sents:

                        text = " ".join(sent["sent_tokens"])
                        passage_offset_end = passage_offset_start + len(text)

                        passages.append(
                            {
                                "id": str(sent["sent_id"]),
                                "type": "",
                                "text": [text],
                                "offsets": [passage_offset_start, passage_offset_end],
                            }
                        )
                        passage_offset_start = passage_offset_end

                    new_entities = []
                    for entity_sent in line["sents"]:
                        entities = entity_sent["entities"]
                        sent_id = entity_sent["sent_id"]
                        sent_offsets, sent_types, sent_texts = [], [], []

                        for entity in entities:
                            sent_offsets.append([entity["start"], entity["end"]])
                            sent_types.append(entity["type"])
                            sent_texts.append(entity["text"])

                        new_entities.append(
                            {
                                "id": sent_id,
                                "type": sent_types,
                                "text": sent_texts,
                                "offsets": sent_offsets,
                                "normalized": [
                                    {
                                        "db_name": "",
                                        "db_id": "",
                                    }
                                ],
                            }
                        )

                    yield key, {
                        "id": str(key),
                        "document_id": line["doc_id"],
                        "passages": passages,
                        "entities": new_entities,
                        "events": [],
                        "coreferences": [],
                        "relations": [],
                    }


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
