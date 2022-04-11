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

"""

import os
from typing import List, Tuple, Dict
from pathlib import Path

import datasets
from utils import schemas, parsing
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{verspoor2013annotating,
title={Annotating the biomedical literature for the human variome},
author={Verspoor, Karin and Jimeno Yepes, Antonio and Cavedon, Lawrence and McIntosh, Tara and Herten-Crabb, Asha and Thomas, Zo{"e} and Plazzer, John-Paul},
journal={Database},
volume={2013},
year={2013},
publisher={Oxford Academic}
}
"""

_DATASETNAME = "verspoor_2013"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset is designed for XXX NLP task.
"""

_HOMEPAGE = ""

# The source repo (https://github.com/rockt/SETH/blob/master/LICENSE) is license
# Apache 2.0 but does that match the dataset? Best info I have for now...
_LICENSE = """\
Copyright 2013 Humboldt-Universität zu Berlin, Dept. of Computer Science and Dept.
of Wissensmanagement in der Bioinformatik

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

_URLS = ['http://github.com/rockt/SETH/zipball/master/']

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.RELATION_EXTRACTION,
]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class Verspoor2013Dataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="verspoor_2013_source",
            version=SOURCE_VERSION,
            description="verspoor_2013 source schema",
            schema="source",
            subset_id="verspoor_2013",
        ),
        BigBioConfig(
            name="verspoor_2013_bigbio_kb",
            version=BIGBIO_VERSION,
            description="verspoor_2013 BigBio schema",
            schema="bigbio_kb",
            subset_id="verspoor_2013",
        ),
    ]

    DEFAULT_CONFIG_NAME = "verspoor_2013_source"

    _ENTITY_TYPES = {
        "Concepts_Ideas",
        "Disorder",
        "Phenomena",
        "Physiology",
        "age",
        "body-part",
        "cohort-patient",
        "disease",
        "ethnicity",
        "gender",
        "gene",
        "mutation",
        "size",
    }

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "events": [  # E line in brat
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
                    ],
                    "relations": [  # R line in brat
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
                    ],
                    "equivalences": [  # Equiv line in brat
                        {
                            "id": datasets.Value("string"),
                            "ref_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                    "attributes": [  # M or A lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                    "normalizations": [  # N lines in brat
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
                    ],
                },
            )

        # Choose the appropriate bigbio schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            # description=_DESCRIPTION,
            features=features,
            # homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

       
        data_dir = Path(dl_manager.download_and_extract(_URLS))

        verspoor_dir = list(data_dir.glob('*/resources/Verspoor2013'))[0]
        text_files = list(verspoor_dir.glob('corpus/*.txt'))
        annotation_dir = verspoor_dir / 'annotations'
        
        data_files = {'text_files': text_files, 'annotation_dir': annotation_dir}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_files": data_files,
                    "split": "train",
                },
            )
        ]

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, data_files, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
       
        if self.config.schema == "source":
            txt_files = data_files['text_files']
            annotation_dir = data_files['annotation_dir']
            for guid, txt_file in enumerate(txt_files):
                example = parsing.parse_brat_file(txt_file, annotation_dir)
                example["id"] = str(guid)
                yield guid, example

        elif self.config.schema == "bigbio_[bigbio_schema_name]":
            txt_files = list(data_files.glob("*txt"))
            annotation_dir = data_files['annotation_dir']
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file), entity_types=self._ENTITY_TYPES
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
