
   
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

import datasets
from datasets import load_dataset


_DATASETNAME = "biosses"

_CITATION = """\
@article{souganciouglu2017biosses,
  title={BIOSSES: a semantic sentence similarity estimation system for the biomedical domain},
  author={So{\u{g}}anc{\i}o{\u{g}}lu, Gizem and {\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan},
  journal={Bioinformatics},
  volume={33},
  number={14},
  pages={i49--i58},
  year={2017},
  publisher={Oxford University Press}
}
"""

_DESCRIPTION = """\
BioSSES computes similarity of biomedical sentences by utilizing WordNet as the general domain  ontology 
and UMLS as the biomedical domain specific ontology.
"""

_HOMEPAGE = "https://tabilab.cmpe.boun.edu.tr/BIOSSES/SourceCode.html"

_LICENSE = ""

_URLs = {"biosses": "https://raw.githubusercontent.com/debajyotidatta/biomedical/biosses_add/examples/biosses/biosses_full.tsv"}

_VERSION = "1.0.0"


class MQPDataset(datasets.GeneratorBasedBuilder):
    """BIOSSES : Biomedical Semantic Similarity Estimation System"""

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

    ['sentence_id', 'sentence_1', 'sentence_2', 'annotator_a', 'annotator_b',
       'annotator_c', 'annotator_d', 'annotator_e']

    def _info(self):

        if self.config.name == _DATASETNAME:
            features = datasets.Features(
                {
                    "sentence_id": datasets.Value("int64"),
                    "sentence_1": datasets.Value("string"),
                    "sentence_2": datasets.Value("string"),
                    "annotator_a": datasets.Value("int64"),
                    "annotator_b": datasets.Value("int64"),
                    "annotator_c": datasets.Value("int64"),
                    "annotator_d": datasets.Value("int64"),
                    "annotator_e": datasets.Value("int64")
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
        ds_dict = load_dataset('csv', delimiter='\t', encoding='utf-8',
                                column_names=['sentence_id', 'sentence_1', 'sentence_2', 
                                'annotator_a', 'annotator_b','annotator_c', 'annotator_d', 'annotator_e'],
                               data_files=filepath)

        if self.config.name == _DATASETNAME:
            for id_, (split, dataset) in enumerate(ds_dict.items()):
                yield id_, {
                    "sentence_id": dataset['sentence_id'][id_],
                    "sentence_1": dataset['sentence_1'][id_],
                    "sentence_2": dataset['sentence_2'][id_],
                    "annotator_a": dataset['annotator_a'][id_],
                    "annotator_b": dataset['annotator_b'][id_],
                    "annotator_c": dataset['annotator_c'][id_],
                    "annotator_d": dataset['annotator_d'][id_],
                    "annotator_e": dataset['annotator_e'][id_]
                }