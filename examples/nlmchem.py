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

import os
import re

import bioc
import datasets

_CITATION = """\
@Article{islamaj2021nlm,
title={NLM-Chem, a new resource for chemical entity recognition in PubMed full text literature},
author={Islamaj, Rezarta and Leaman, Robert and Kim, Sun and Kwon, Dongseop and Wei, Chih-Hsuan and Comeau, Donald C and Peng, Yifan and Cissel, David and Coss, Cathleen and Fisher, Carol and others},
journal={Scientific Data},
volume={8},
number={1},
pages={1--12},
year={2021},
publisher={Nature Publishing Group}
}
"""

_DESCRIPTION = """\
NLM-Chem corpus consists of 150 full-text articles from the PubMed Central Open Access dataset,
comprising 67 different chemical journals, aiming to cover a general distribution of usage of chemical
names in the biomedical literature.
Articles were selected so that human annotation was most valuable (meaning that they were rich in bio-entities,
and current state-of-the-art named entity recognition systems disagreed on bio-entity recognition.
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-2"

_LICENSE = " CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"

# files found here `https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/` have issues at extraction
# _URLs = {"biocreative": "https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem" }
_URLs = {
    "biocreative": "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-NLMChem-corpus_v2.BioC.xml.gz"
}


class NLMChemDataset(datasets.GeneratorBasedBuilder):
    """NLMChem"""

    # v2 here: https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/
    VERSION = datasets.Version("1.0.0")

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

    DEFAULT_CONFIG_NAME = "biocreative"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):

        if self.config.name == "biocreative":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "passages": datasets.Sequence(
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ),
                    "entities": datasets.Sequence(
                        {
                            "id": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence([datasets.Value("string")]),
                            "type": datasets.Value("string"),
                            "normalized": datasets.Sequence(
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ),
                        }
                    ),
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
                        data_dir, "BC7T2-NLMChem-corpus-train.BioC.xml"
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "BC7T2-NLMChem-corpus-test.BioC.xml"
                    ),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "BC7T2-NLMChem-corpus-dev.BioC.xml"
                    ),
                    "split": "dev",
                },
            ),
        ]

    def _get_passages_and_entities(
        self, d: bioc.BioCDocument
    ) -> tuple[list[dict], list[dict]]:

        passages: list[dict] = []
        entities: list[dict] = []

        text_total_length = 0

        for id_, p in enumerate(d.passages):

            offset = p.offset - text_total_length

            length = offset + len(p.text)

            text_total_length += len(p.text) + 1

            ep = {
                "id": str(id_),
                "text": p.text,
                "type": p.infons.get("type"),
                "offsets": [(offset, length)],
            }

            passages.append(ep)

            for a in p.annotations:

                # atype = a.infons.get("type")
                #
                # # no in-text annotation
                # if entity_type in ["MeSH_Indexing_Chemical", "OTHER"]:
                #     continue

                ea = {
                    "type": a.infons.get("type"),
                    "offsets": [
                        (loc.offset - offset, loc.offset + loc.length - offset)
                        for loc in a.locations
                    ],
                    "text": [[a.text]],
                    "id": a.id,
                    "normalized": self._get_normalized(a),
                }

                entities.append(ea)

        return passages, entities

    def _get_normalized(self, a: bioc.BioCAnnotation) -> list[dict]:
        """
        Get normalization DB and ID from annotation
        """

        identifiers = a.infons.get("identifier")

        if identifiers is not None:

            identifiers = re.split(r",|;", identifiers)

            identifiers = [i for i in identifiers if i != "-"]

            normalized = [i.split(":") for i in identifiers]

            normalized = [
                {"db_name": elems[0], "db_id": elems[1]} for elems in normalized
            ]

        else:

            normalized = [{"db_name": "-1", "db_id": "-1"}]

        return normalized

    def _generate_examples(
        self,
        filepath: str,
        split: str,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""
        if self.config.name == "biocreative":
            reader = bioc.BioCXMLDocumentReader(str(filepath))
            for did, doc in enumerate(reader):

                passages, entities = self._get_passages_and_entities(doc)
                encoded = {
                    "id": str(did),
                    "document_id": doc.id,
                    "passages": passages,
                    "entities": entities,
                }

                yield did, encoded


# def _get_entities(self, d: bioc.BioCDocument) -> list[dict]:
#     """
#     Get all normalized entities in document
#     """
#
#     return [
#         {
#             "id": a.id,
#             "offsets": [
#                 (loc.offset, loc.offset + loc.length) for loc in a.locations
#             ],
#             "text": [a.text],
#             "type": a.infons.get("type"),
#             "normalized": self._get_normalized(a),
#         }
#         for p in d.passages
#         for a in p.annotations
#     ]
#
# def _get_passages(self, d: bioc.BioCDocument) -> list[dict]:
#     """
#     Encode passages in schema
#     """
#
#     offset_start = 0
#     passages = []
#
#     for id_, p in enumerate(d.passages):
#
#         offset_end = offset_start + len(p.text)
#
#         encoded = {
#             "id": str(id_),
#             "type": p.infons.get("type"),
#             "text": p.text,
#             "offsets": [(offset_start, offset_end)],
#         }
#
#         # white space separating passages
#         offset_start = offset_end + 1
#
#         passages.append(encoded)
#
#     return passages
