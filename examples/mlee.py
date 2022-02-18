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
from collections import defaultdict
from pathlib import Path

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

_DATASETNAME = "mlee"
_SOURCE_VIEW_NAME = "source"
_UNIFIED_VIEW_NAME = "bigbio"

_CITATION = """\
@article{,
  author = {Pyysalo, Sampo and Ohta, Tomoko and Miwa, Makoto and Cho, Han-Cheol and Tsujii, Jun'ichi and Ananiadou, Sophia},
  title = "{Event extraction across multiple levels of biological organization}",
  journal   = {Bioinformatics},
  volume    = {28},
  year      = {2012},
  url       = {https://doi.org/10.1093/bioinformatics/bts407},
  doi       = {10.1093/bioinformatics/bts407},
  biburl    = {},
  bibsource = {}
}
"""

_DESCRIPTION = """\
A description of your dataset
"""

_HOMEPAGE = "http://www.nactem.ac.uk/MLEE/"

_LICENSE = "CC BY-NC-SA 3.0"

_URLs = {"mlee": "http://www.nactem.ac.uk/MLEE/MLEE-1.0.2-rev1.tar.gz"}

_VERSION = "1.0.0"

"""
Step 3: Change the class name to correspond to your <Your_Dataset_Name> 
ex: "ChemProtDataset".

Then, fill all relevant information to `BuilderConfig` which populates information about the class. You may have multiple builder configs (ex: a large dataset separated into multiple partitions) if you populate for different dataset names + descriptions. The following is setup for just 1 dataset, but can be adjusted.

NOTE - train/test/dev splits can be handled in `_split_generators`.
"""


class MLEE(datasets.GeneratorBasedBuilder):
    """Write a short docstring documenting what this dataset is"""

    VERSION = datasets.Version(_VERSION)

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_SOURCE_VIEW_NAME,
            version=VERSION,
            description=_DESCRIPTION,
        ),
        datasets.BuilderConfig(
            name=_UNIFIED_VIEW_NAME,
            version=VERSION,
            description="BigScience biomedical hackathon schema",
        ),
    ]

    DEFAULT_CONFIG_NAME = _SOURCE_VIEW_NAME

    _ENTITY_TYPES = {
        "Anatomical_system",
        "Cell",
        "Cellular_component",
        "DNA_domain_or_region",
        "Developing_anatomical_structure",
        "Drug_or_compound",
        "Gene_or_gene_product",
        "Immaterial_anatomical_entity",
        "Multi-tissue_structure",
        "Organ",
        "Organism",
        "Organism_subdivision",
        "Organism_substance",
        "Pathological_formation",
        "Protein_domain_or_region",
        "Tissue",
    }

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

        if self.config.name == _SOURCE_VIEW_NAME:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": datasets.Sequence(  # T line in brat, e.g. type or event trigger
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ),
                    "events": datasets.Sequence(  # E line in brat
                        {
                            "trigger": datasets.Value(
                                "string"
                            ),  # refers to the text_bound_annotation of the trigger,
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arguments": datasets.Sequence(
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ),
                        }
                    ),
                    "relations": datasets.Sequence(  # R line in brat
                        {
                            "id": datasets.Value("string"),
                            "head": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "tail": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "type": datasets.Value("string"),
                        }
                    ),
                    "equivalences": datasets.Sequence(  # Equiv line in brat
                        {
                            "id": datasets.Value("string"),
                            "ref_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ),
                    "attributes": datasets.Sequence(  # M or A lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ),
                    "normalizations": datasets.Sequence(  # N lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value(
                                "string"
                            ),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value(
                                "string"
                            ),  # ID in the resource, e.g. 534366
                            "text": datasets.Value(
                                "string"
                            ),  # Human readable description/name of the entity, e.g. "Barack Obama"
                        }
                    ),
                },
            )
        elif self.config.name == _UNIFIED_VIEW_NAME:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "passages": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": datasets.Sequence(datasets.Value("int32")),
                        }
                    ],
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "events": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            # refers to the text_bound_annotation of the trigger
                            "trigger": {
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                                "text": datasets.Value("string")
                            },
                            "arguments": [
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "coreferences": [
                        {
                            "id": datasets.Value("string"),
                            "entity_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
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

        my_urls = _URLs[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": data_dir
            / "MLEE-1.0.2-rev1"
            / "standoff"
            / "development"
            / "train",
            "dev": data_dir / "MLEE-1.0.2-rev1" / "standoff" / "development" / "test",
            "test": data_dir / "MLEE-1.0.2-rev1" / "standoff" / "test" / "test",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_files": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_files": data_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_files": data_files["test"]},
            ),
        ]

    def _generate_examples(self, data_files: Path):
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
        if self.config.name == _SOURCE_VIEW_NAME:
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = self.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example
        elif self.config.name == _UNIFIED_VIEW_NAME:
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = self.brat_parse_to_unified_schema(
                    self.parse_brat_file(txt_file)
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

    def brat_parse_to_unified_schema(self, brat_parse: Dict) -> Dict:
        unified_example = {}

        # Prefix all ids with document id to ensure global uniqueness,
        # because brat ids are only unique within their document
        id_prefix = brat_parse["document_id"] + "_"

        # identical
        unified_example["document_id"] = brat_parse["document_id"]
        unified_example["passages"] = [
            {
                "id": id_prefix + "_text",
                "type": "abstract",
                "text": brat_parse["text"],
                "offsets": [0, len(brat_parse["text"])],
            }
        ]

        # get normalizations
        ref_id_to_normalizations = defaultdict(list)
        for normalization in brat_parse["normalizations"]:
            ref_id_to_normalizations[normalization["ref_id"]].append(
                {
                    "db_name": normalization["resource_name"],
                    "db_id": normalization["cuid"],
                }
            )

        # separate entities and event triggers
        unified_example["entities"] = []
        id_to_event_trigger = {}
        for ann in brat_parse["text_bound_annotations"]:
            if ann["type"] in self._ENTITY_TYPES:
                entity_ann = ann.copy()
                entity_ann["id"] = id_prefix + entity_ann["id"]
                entity_ann["normalized"] = ref_id_to_normalizations[ann["id"]]
                unified_example["entities"].append(entity_ann)
            else:
                id_to_event_trigger[ann["id"]] = ann

        unified_example["events"] = []
        for event in brat_parse["events"]:
            event = event.copy()
            event["id"] = id_prefix + event["id"]
            trigger = id_to_event_trigger[event["trigger"]]
            event["trigger"] = {
                "text": trigger["text"].copy(),
                "offsets": trigger["offsets"].copy(),
            }
            for argument in event["arguments"]:
                argument["ref_id"] = id_prefix + argument["ref_id"]

            unified_example["events"].append(event)

        # massage relations
        unified_example["relations"] = []
        for ann in brat_parse["relations"]:
            unified_example["relations"].append(
                {
                    "arg1_id": id_prefix + ann["head"]["ref_id"],
                    "arg2_id": id_prefix + ann["tail"]["ref_id"],
                    "id": id_prefix + ann["id"],
                    "type": ann["type"],
                    "normalized": [],
                }
            )

        # get coreferences
        unified_example["coreferences"] = []
        for i, ann in enumerate(brat_parse["equivalences"], start=1):
            is_entity_cluster = True
            for ref_id in ann["ref_ids"]:
                if not ref_id.startswith("T"):  # not textbound -> no entity
                    is_entity_cluster = False
                elif ref_id in id_to_event_trigger:  # event trigger -> no entity
                    is_entity_cluster = False
            if is_entity_cluster:
                entity_ids = [id_prefix + i for i in ann["ref_ids"]]
                unified_example["coreferences"].append(
                    {"id": id_prefix + str(i), "entity_ids": entity_ids}
                )

        return unified_example

    def parse_brat_file(self, txt_file):
        example = {}
        example["document_id"] = txt_file.name.removesuffix(".txt")
        with txt_file.open() as f:
            example["text"] = f.read()
        a1_file = txt_file.with_suffix(".a1")
        a2_file = txt_file.with_suffix(".a2")
        ann_lines = []
        with a1_file.open() as f:
            ann_lines.extend(f.readlines())
        with a2_file.open() as f:
            ann_lines.extend(f.readlines())
        example["text_bound_annotations"] = []
        example["events"] = []
        example["relations"] = []
        example["equivalences"] = []
        example["attributes"] = []
        example["normalizations"] = []
        for line in ann_lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("T"):  # Text bound
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["text"] = [fields[2]]
                ann["type"] = fields[1].split()[0]
                ann["offsets"] = []
                span_str = fields[1].removeprefix(ann["type"] + " ")
                for span in span_str.split(";"):
                    start, end = span.split()
                    ann["offsets"].append([int(start), int(end)])

                example["text_bound_annotations"].append(ann)

            elif line.startswith("E"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                ann["type"], ann["trigger"] = fields[1].split()[0].split(":")

                ann["arguments"] = []
                for role_ref_id in fields[1].split()[1:]:
                    argument = {
                        "role": (role_ref_id.split(":"))[0],
                        "ref_id": (role_ref_id.split(":"))[1],
                    }
                    ann["arguments"].append(argument)

                example["events"].append(ann)

            elif line.startswith("R"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["type"] = fields[1].split()[0]

                ann["head"] = {
                    "role": fields[1].split()[1].split(":")[0],
                    "ref_id": fields[1].split()[1].split(":")[1],
                }
                ann["tail"] = {
                    "role": fields[1].split()[2].split(":")[0],
                    "ref_id": fields[1].split()[2].split(":")[1],
                }

                example["relations"].append(ann)

            # '*' seems to be the legacy way to mark equivalences,
            # but I couldn't find any info on the current way
            # this might have to be adapted dependent on the brat version
            # of the annotation
            elif line.startswith("*"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["ref_ids"] = fields[1].split()[1:]

                example["equivalences"].append(ann)

            elif line.startswith("A") or line.startswith("M"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                info = fields[1].split()
                ann["type"] = info[0]
                ann["ref_id"] = info[1]

                if len(info) > 2:
                    ann["value"] = info[2]
                else:
                    ann["value"] = ""

                example["attributes"].append(ann)

            elif line.startswith("N"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["text"] = fields[2]

                info = fields[1].split()

                ann["type"] = info[0]
                ann["ref_id"] = info[1]
                ann["resource_name"] = info[2].split(":")[0]
                ann["cuid"] = info[2].split(":")[1]
                example["normalizations"].append(ann)

        return example
