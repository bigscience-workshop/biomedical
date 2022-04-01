# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
#
# * <append your name and optionally your github handle here>
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

"""

import os

from typing import List

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{nentidis-etal-2017-results,
  author    = {Nentidis, Anastasios  and Bougiatiotis, Konstantinos  and Krithara, Anastasia  and
               Paliouras, Georgios  and Kakadiaris, Ioannis},
  title     = {Results of the fifth edition of the {B}io{ASQ} Challenge},
  journal   = {},
  volume    = {BioNLP 2017},
  year      = {2007},
  url       = {https://aclanthology.org/W17-2306},
  doi       = {10.18653/v1/W17-2306},
  biburl    = {},
  bibsource = {https://aclanthology.org/W17-2306}
}
"""

# TODO: create a module level variable with your dataset name (should match script name)
_DATASETNAME = "bioasq_task_c_2017"

_DESCRIPTION = """\
The training data set for this task contains annotated biomedical articles published in PubMed and corresponding full text from PMC. By annotated is meant that GrantIDs and corresponding Grant Agencies have been identified in the full text of articles.
"""

_HOMEPAGE = "http://participants-area.bioasq.org/general_information/Task5c/"

_LICENSE = "NLM License Code: 8283NLM123"

# TODO: Add links to the urls needed to download your dataset files.
# For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: "https://ii.nlm.nih.gov/BioASQ/Task5C/BioASQ_2017_Task5C_Training.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

# this version doesn't have to be consistent with semantic versioning. Anything that is
# provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BioASQTaskC2017(datasets.GeneratorBasedBuilder):
    """
    BioASQ Task C on 
    """

    DEFAULT_CONFIG_NAME = "bioasq_task_c_2017_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bioasq_task_c_2017_source",
            version=SOURCE_VERSION,
            description="bioasq_task_c_2017 source schema",
            schema="source",
            subset_id="bioasq_task_c_2017",
        ),
        BigBioConfig(
            name="bioasq_task_c_2017_bigbio_[schema_name]",
            version=BIGBIO_VERSION,
            description="bioasq_task_c_2017 BigBio schema",
            schema="bigbio_[bigbio_schema_name]",
            subset_id="bioasq_task_c_2017",
        )
    ]


    def _info(self) -> datasets.DatasetInfo:
        
        # BioASQ Task C source schema
        if self.config.schema == "source":
            features = datasets.Features(
               {
                   "pmid": datasets.Value("string"),
                   "pmcid": datasets.Value("string"),
                   "grantList": [
                       {
                           "grantID": datasets.Value("string"),
                           "agency": datasets.Value("string"),
                       }
                   ],
               }
            )

        # Choose the appropriate bigbio schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema =="bigbio_[bigbio_schema_name]":
            # e.g. features = schemas.kb_features
            # TODO: Choose your big-bio schema here
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "bigbio" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument. Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        # TODO: KEEP if your dataset is PUBLIC; REMOVE if not
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        # TODO: KEEP if your dataset is LOCAL; remove if NOT
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits. If your dataset does not have any splits, you can omit any missing splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath, split) -> (int, dict):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files

        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            for key, example in thing:
                yield key, example

        elif self.config.schema == "bigbio":
            # TODO: yield (key, example) tuples in the bigbio schema
            for key, example in thing:
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
