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

import json
import os
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {Jain, S., Agrawal, A., Saporta, A., Truong, S. Q., Nguyen Duong, D., Bui, T., Chambon, P., Lungren, M., Ng, A., Langlotz, C., & Rajpurkar, P. },
  title     = {RadGraph: Extracting Clinical Entities and Relations from Radiology Reports (version 1.0.0)},
  journal   = {PhysioNet},
  volume    = {},
  year      = {2021},
  url       = {https://physionet.org/content/radgraph/1.0.0/},
  doi       = {10.13026/hm87-5p47},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "radgraph"

_DESCRIPTION = """\
This dataset is derived from radiology reports and is designed for named entity recognition and relatation extraction.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = "https://physionet.org/content/radgraph/1.0.0/"

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = """\
    The PhysioNet Credentialed Health Data License
    Version 1.5.0

    Copyright (c) 2022 MIT Laboratory for Computational Physiology

    The MIT Laboratory for Computational Physiology (MIT-LCP) wishes to make data available for research and educational purposes to qualified requestors, but only if the data are used and protected in accordance with the terms and conditions stated in this License.

    It is hereby agreed between the data requestor, hereinafter referred to as the "LICENSEE", and MIT-LCP, that:

    The LICENSEE will not attempt to identify any individual or institution referenced in PhysioNet restricted data.
    The LICENSEE will exercise all reasonable and prudent care to avoid disclosure of the identity of any individual or institution referenced in PhysioNet restricted data in any publication or other communication.
    The LICENSEE will not share access to PhysioNet restricted data with anyone else.
    The LICENSEE will exercise all reasonable and prudent care to maintain the physical and electronic security of PhysioNet restricted data.

    If the LICENSEE finds information within PhysioNet restricted data that he or she believes might permit identification of any individual or institution, the LICENSEE will report the location of this information promptly by email to PHI-report@physionet.org, citing the location of the specific information in question.
    The LICENSEE will use the data for the sole purpose of lawful use in scientific research and no other.
    The LICENSEE will be responsible for ensuring that he or she maintains up to date certification in human research subject protection and HIPAA regulations.
    The LICENSEE agrees to contribute code associated with publications arising from this data to a repository that is open to the research community.
    This agreement may be terminated by either party at any time, but the LICENSEE's obligations with respect to PhysioNet data shall continue after termination.  
    THE DATA ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA OR THE USE OR OTHER DEALINGS IN THE DATA.
    """

# Local dataset - available only after completing PhysioNet requirements
_URLS = {}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION, 
    Tasks.RELATION_EXTRACTION
    ] 

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class RadgraphDataset(datasets.GeneratorBasedBuilder):
    """RadGraph is a dataset of entities and relations in full-text radiology reports."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="radgraph_source",
            version=SOURCE_VERSION,
            description="RadGraph source schema",
            schema="source",
            subset_id="radgraph",
        ),
        BigBioConfig(
            name="radgraph_bigbio_kb",
            version=BIGBIO_VERSION,
            description="RadGraph BigBio schema",
            schema="bigbio_kb",
            subset_id="radgraph",
        ),
    ]

    DEFAULT_CONFIG_NAME = "radgraph_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features( 
                {
                    "report_id" : {
                        "text": datasets.Value("string"),
                        "entities": {
                            "entity_id": {
                                "tokens": datasets.Value("string"),
                                "label": datasets.Value("string"),
                                "start_ix": datasets.Value("int32"),
                                "end_ix": datasets.Value("int32"),
                                "relations": [[datasets.Value("string")]]
                            },
                        },
                        "data_source": datasets.Value("string"),
                        "data_split": datasets.Value("string"),
                    }
                }
            )

        # Choose the appropriate bigbio schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "bigbio_kb":
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
        """Returns SplitGenerators."""

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.json"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.json"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files

        '''
        "report_id": {
                        "text": datasets.Value("string"),
                        "entities": {
                            "entity_id": {
                                "tokens": datasets.Value("string"),
                                "label": datasets.Value("string"),
                                "start_ix": datasets.Value("int32"),
                                "end_ix": datasets.Value("int32"),
                                "relations": [[datasets.Value("string")]]
                            },
                        },
                        "data_source": datasets.Value("string"),
                        "data_split": datasets.Value("string"),
                    }
        '''

        if self.config.schema == "source":
            with open(filepath) as json_file:
                data = json.load(json_file)
                uid = 0
                for chart_id in data:
                    example = {}
                    chart_data = data[chart_id]
                    example = { 
                        chart_id: {
                            "text" : chart_data["text"],
                            "entities" : chart_data["entities"],
                            "data_source" : chart_data["data_source"],
                            "data_split": chart_data["data_split"]
                        }
                    }
                    yield uid, example
                    uid +=1

        # elif self.config.schema == "bigbio_kb":
        #     # TODO: yield (key, example) tuples in the bigbio schema
        #     for key, example in thing:
        #         yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
#if __name__ == "__main__":
#    datasets.load_dataset(__file__)
