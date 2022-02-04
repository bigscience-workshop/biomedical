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
"""This script was originally authored by Jason Fries"""


import csv
import json
import os

import bioc
import datasets

_CITATION = """\
@article{DBLP:journals/biodb/LiSJSWLDMWL16,
  author    = {Jiao Li and
               Yueping Sun and
               Robin J. Johnson and
               Daniela Sciaky and
               Chih{-}Hsuan Wei and
               Robert Leaman and
               Allan Peter Davis and
               Carolyn J. Mattingly and
               Thomas C. Wiegers and
               Zhiyong Lu},
  title     = {BioCreative {V} {CDR} task corpus: a resource for chemical disease
               relation extraction},
  journal   = {Database J. Biol. Databases Curation},
  volume    = {2016},
  year      = {2016},
  url       = {https://doi.org/10.1093/database/baw068},
  doi       = {10.1093/database/baw068},
  timestamp = {Thu, 13 Aug 2020 12:41:41 +0200},
  biburl    = {https://dblp.org/rec/journals/biodb/LiSJSWLDMWL16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
The BioCreative V Chemical Disease Relation (CDR) dataset is a large annotated text corpus of 
human annotations of all chemicals, diseases and their interactions in 1,500 PubMed articles.
"""

_HOMEPAGE = "http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/"

_LICENSE = "Public Domain Mark 1.0"

_URLs = {"biocreative": "http://www.biocreative.org/media/store/files/2016/CDR_Data.zip"}


class Bc5cdrDataset(datasets.GeneratorBasedBuilder):
    """BioCreative V Chemical Disease Relation (CDR) Task."""

    VERSION = datasets.Version("01.05.16")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="biocreative",
            version=VERSION,
            description="Original annotation files.",
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        "biocreative"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def _info(self):

        if self.config.name == "biocreative":
            features = datasets.Features(
                {
                    "passages": datasets.Sequence(
                        {
                            "pmid": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "entities": datasets.Sequence(
                                {
                                    "offsets": datasets.Sequence([datasets.Value("int32")]),
                                    "text": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                    "entity_id": datasets.Value("string"),
                                }
                            ),
                            "relations": datasets.Sequence(
                                {
                                    "relation": datasets.Value("string"),
                                    "chemical": datasets.Value("string"),
                                    "disease": datasets.Value("string"),
                                }
                            ),
                        }
                    )
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

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml",
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml",
                    ),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml",
                    ),
                    "split": "dev",
                },
            ),
        ]

    def _get_bioc_entity(self, span, kb_id_key="MESH"):
        offsets = [(loc.offset, loc.offset + loc.length) for loc in span.locations]
        kb_id = span.infons[kb_id_key] if kb_id_key else -1
        return {
            "offsets": offsets,
            "text": span.text,
            "type": span.infons["type"],
            "entity_id": kb_id,
        }

    def _generate_examples(
        self,
        filepath,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""
        if self.config.name == "biocreative":
            reader = bioc.BioCXMLDocumentReader(str(filepath))
            for id_, xdoc in enumerate(reader):
                yield id_, {
                    "passages": [
                        {
                            "pmid": xdoc.id,
                            "type": passage.infons["type"],
                            "text": passage.text,
                            "entities": [self._get_bioc_entity(span) for span in passage.annotations],
                            "relations": [
                                {key.lower(): value for key, value in rel.infons.items()} for rel in xdoc.relations
                            ],
                        }
                        for passage in xdoc.passages
                    ]
                }
