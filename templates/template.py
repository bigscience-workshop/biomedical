# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
#
# TODO: fill out the line below
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
import datasets


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

# TODO: create a module level variable with your dataset name (should match script name)
_DATASETNAME = "dataset_name"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset is designed for XXX NLP task.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here (if possible)
_LICENSE = ""

# TODO: Add links to the urls needed to download your dataset files.
# For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: "url or list of urls or ... ",
}

# TODO: set this to a version that is associated with the dataset. if none exists use 1.0.0
_SOURCE_VERSION = ""

_BIGBIO_VERSION = "1.0.0"

# TODO: Set the NLP task here
# Options include: "kb", "qa", "entailment", "text_to_text", "text", "pairs"
_SUPPORTED_TASKS = ["your_task_here"]

# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="source", version=VERSION, description="Source schema"
        ), # TODO [Optional]: Source of dataset
        datasets.BuilderConfig(
            name="bigbio",
            version=BIGBIO_VERSION,
            description="BigScience Biomedical schema",
        ), # TODO: Big-bio schema config
    ]

    DEFAULT_CONFIG_NAME = "source"

    def _info(self):

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.name == "source":
            # TODO: Create your source schema here
            raise NotImplementedError()

            # EX: Arbitrary NER type dataset
            #features = datasets.Features(
            #    {
            #        "doc_id": datasets.Value("string"),
            #        "text": datasets.Value("string"),
            #        "entities": [
            #            {
            #                "offsets": [datasets.Value("int64")],
            #                "text": datasets.Value("string"),
            #                "type": datasets.Value("string"),
            #                "entity_id": datasets.Value("string"),
            #            }
            #        ],
            #    }
            #)

        # Choose the appropriate bigbio schema for your task and copy it here. You can find the big-bio schemas here: https://github.com/bigscience-workshop/biomedical/tree/master/schemas or in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks. In that case you can define multiple bigbio configs with a bigbio-<task> format.

        # For example bigbio-translation, bigbio-ner
        elif self.config.name == "bigbio":
            # TODO: Implement your big-bio schema here
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
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
        data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits. If your dataset does not have any splits, you can omit any missing splits.

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

    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files 

        if self.config.name == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            for key, example in thing:
                yield key, example

        elif self.config.name == "bigbio":
            # TODO: yield (key, example) tuples in the bigbio schema
            for key, example in thing:
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py