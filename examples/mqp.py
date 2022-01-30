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

import os
from typing import Dict, Tuple
import datasets
from datasets import load_dataset


_DATASETNAME = "mqp"

_CITATION = """\
@article{DBLP:journals/biodb/LiSJSWLDMWL16,
  author    = {Krallinger, M., Rabal, O., Lourenço, A.},
  title     = {Effective Transfer Learning for Identifying Similar Questions: Matching User Questions to COVID-19 FAQs},
  journal   = {KDD '20: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  volume    = {3458–3465},
  year      = {2020},
  url       = {https://github.com/curai/medical-question-pair-dataset},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DESCRIPTION = """\
Medical Question Pairs dataset by McCreery et al (2020) contains pairs of medical questions and paraphrased versions of 
the question prepared by medical professional. Paraphrased versions were labelled as similar (syntactically dissimilar 
but contextually similar ) or dissimilar (syntactically may look similar but contextually dissimilar). Labels 1: similar, 0: dissimilar
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/"

_LICENSE = ""

_URLs = {"mqp": "https://raw.githubusercontent.com/curai/medical-question-pair-dataset/master/mqp.csv"}

_VERSION = "1.0.0"


class MQPDataset(datasets.GeneratorBasedBuilder):
    """BioCreative VI Chemical-Protein Interaction Task."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DATASETNAME,
            version=VERSION,
            description=_DESCRIPTION,
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        _DATASETNAME  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def _info(self):

        if self.config.name == _DATASETNAME:
            features = datasets.Features(
                {
                    "annot_id": datasets.Value("int64"),
                    "sentence_1": datasets.Value("string"),
                    "sentence_2": datasets.Value("string"),
                    "relation": datasets.Value("int64")
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
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

    def _split_generators(self):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        my_urls = _URLs[self.config.name]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": my_urls,
                    "split": "train",
                }
            )
        ]

    def _generate_examples(self, filepath,  split):
        """Yields examples as (key, example) tuples."""
        ds_dict = load_dataset('csv', delimiter=',',
                                column_names=['annot_id', 'sentence_1', 'sentence_2', 'relation'],
                               data_files=filepath)

        if self.config.name == _DATASETNAME:
            for id_, (split, dataset) in enumerate(ds_dict.items()):
                yield id_, {
                    "annot_id": dataset['annot_id'][id_],
                    "sentence_1": dataset['sentence_1'][id_],
                    "sentence_2": dataset['sentence_2'][id_],
                    "relation": dataset['relation'][id_],
                }








