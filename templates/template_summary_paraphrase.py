# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
This is a template on how to implement a dataset in the biomedical repo.

A thorough walkthrough on how to implement a dataset can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

This script corresponds to Step 4 in the Biomedical Hackathon guide.

To start, copy this template file and save it as <your_dataset_name>.py in an appropriately named folder within datasets. Then, modify this file as necessary to implement your own method of extracting, and generating examples for your dataset. 

There are 3 key elements to implementing a dataset:

(1) `_info`: Create a skeletal structure that describes what is in the dataset and the nature of the features.

(2) `_split_generators`: Download and extract data for each split of the data (ex: train/dev/test)

(3) `_generate_examples`: From downloaded + extracted data, process files for the data in a feature format specified in "info".

----------------------
Step 1: Declare imports
Your imports go here; the only mandatory one is `datasets`, as methods and attributes from this library will be used throughout the script.

We have provided some import statements that we strongly recommend. Feel free to adapt; so long as the style-guide requirements are satisfied (Step 5), then you should be able to push your code.
"""
import datasets
import os  # useful for paths
from typing import Iterable, Dict, List
import logging


"""
Step 2: Create keyword descriptors for your dataset

The following variables are used to populate the dataset entry. Common ones include:

- `_DATASETNAME` = "your_dataset_name"
- `_CITATION`: Latex-style citation of the dataset
- `_DESCRIPTION`: Explanation of the dataset
- `_HOMEPAGE`: Where to find the dataset's hosted location
- `_LICENSE`: License to use the dataset
- `_URLs`: How to download the dataset(s), by name; make this in the form of a dictionary where <dataset_name> is the key and <url_of_dataset> is the value
- `_VERSION`: Version of the dataset
"""

_DATASETNAME = "your_dataset_name"

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

_DESCRIPTION = """\
A description of your dataset
"""

_HOMEPAGE = "Homepage of the dataset"

_LICENSE = "License"

_URLs = {'your_dataset_name': "your_dataset_URL"}

_VERSION = "1.0.0"

"""
Step 3: Change the class name to correspond to your <Your_Dataset_Name> 
ex: "ChemProtDataset".

Then, fill all relevant information to `BuilderConfig` which populates information about the class. You may have multiple builder configs (ex: a large dataset separated into multiple partitions) if you populate for different dataset names + descriptions. The following is setup for just 1 dataset, but can be adjusted.

NOTE - train/test/dev splits can be handled in `_split_generators`.
"""


class YourDatasetName(datasets.GeneratorBasedBuilder):
    """Write a short docstring documenting what this dataset is"""

    VERSION = datasets.Version(_VERSION)

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DATASETNAME,
            version=VERSION,
            description=_DESCRIPTION,
        ),
    ]

    DEFAULT_CONFIG_NAME = _DATASETNAME

    """
    Step 4: Populate "information" about the dataset that creates a skeletal structure for an example within the dataset looks like.

    The following data structures are useful:

    datasets.Features - An instance that defines all descriptors within a feature in an arbitrary nested manner; the "feature" class must strictly adhere to this format. 

    datasets.Value - the type of the data structure (ex: useful for text, PMIDs)

    datasets.Sequence - for information that must be in a continuous sequence (ex: spans in the text, offsets)

    An example is as follows for an ENTITY + RELATION dataset.

    Your format may differ depending on what the dataset is. Please try to keep the extraction as close to the original dataset as possible. If you're having trouble adapting your dataset, please contact the community channels and an organizer will reach out!
    """

    def _info(self):

        if self.config.name == _DATASETNAME:
            features = datasets.Features(
                {
                    "annotator_id": datasets.Value("int64"),
                    "sentence_1": datasets.Value("string"),
                    "sentence_2": datasets.Value("string"),
                    "relation": datasets.Value("int64"),
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """
        Step 5: Download and extract the dataset.

        For each config name, run `download_and_extract` from the dl_manager; this will download and unzip any files within a cache directory, specified by `data_dir`.

        `download_and_extract` can accept an iterable object and return the same structure with the url replaced with the path to local files:

        ex:
        output = dl_manager.download_and_extract({"data1:" "url1", "data2": "url2"})

        output
        >> {"data1": "path1", "data2": "path2"}

        Nested zip files can be cached also, but make sure to save their path.

        Fill the arguments of "SplitGenerator" with `name` and `gen_kwargs`. 

        Note:

        - `name` can be: datasets.Split.<TRAIN/TEST/VALIDATION> or a string
        - all keys in `gen_kwargs` can be passed to `_generate_examples()`. If your dataset has multiple files, you can make a separate key for each file, as shown below:

        """

        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name="DatasetSplit",
                gen_kwargs={
                    "filepath": data_dir,
                    "annotator_file": os.path.join(data_dir, "annotator.txt"),
                    "sentence_1_file": os.path.join(data_dir, "sentence_1.txt"),
                    "sentence_2_file": os.path.join(data_dir, "sentence_2.txt"),
                    "relation_file": os.path.join(data_dir, "relations.txt"),
                    "split": "Name_of_Split",
                },
            ),
        ]

    def _generate_examples(
        self, filepath, annotator_file, sentence_1_file, sentence_2_file, relation_file, split
    ):
        """
        Step 6: Create a generator that yields (key, example) of the dataset of interest.

        The arguments to this function come from `gen_kwargs` returned in `_split_generators()`

        The goal of this function is to perform operations on any of the keys of `gen_kwargs` that allow you to extract and process the data.

        The following skeleton does the following:

        - "extracts" abstracts
        - "extracts" entities, assuming the output is of the form specified in `_info`
        - "extracts" relations, assuming similarly the output in the form specified in `_info`.

        An assumption in this pseudo code is that the abstract, entity, and relation file all have linking keys.
        """
        if self.config.name == _DATASETNAME:

            annotators = self._get_annotators(annotator_file)
            first_sentences = self._get_entities(sentence_1_file)
            second_sentences = self._get_entities(sentence_2_file)
            relations = self._get_relations(relation_file)

            for id_, key in enumerate(annotators):
                yield id_, {
                    "annotator_id": annotators[key],
                    "sentence_1": first_sentences[key],
                    "sentence_2": second_sentences[key],
                    "relation": relations[key],
                }

    @staticmethod
    def _get_annotators(abstract_file: str) -> Dict[int, str]:
        """
        Create a function that can:

        - Read the annotator files
        - Return {key: annotator_id} output.
        """
        pass

    @staticmethod
    def _get_entities(entity_file: str) -> Dict[int, str]:
        """
        Create a function that can:

        - Read the entity file
        - Return a {key: entity}

        Where the entity is a string sentence
        """
        pass

    @staticmethod
    def _get_relation(relation_file: str) -> Dict[int, str]:
        """
        Create a function that can:

        - Read the relation file
        - Return a {key: relation}
        Where relation is a string, designating the type of relation
        """
        pass
