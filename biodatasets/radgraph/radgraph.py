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
The RadGraph dataset is derived from radiology reports and is designed for named entity recognition and relatation extraction.
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
                    "report_id" : datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "entity_id": datasets.Value("string"),
                            "tokens": datasets.Value("string"),
                            "label": datasets.Value("string"),
                            "start_ix": datasets.Value("int32"),
                            "end_ix": datasets.Value("int32"),
                            "labeler": datasets.Value("string"),
                            "relations": [ 
                                {
                                    "relation_id": datasets.Value("string"),
                                    "type": datasets.Value("string"), # e.g. "modify"
                                    "arg": datasets.Value("string") # e.g. "7"
                                },
                            ],
                        },
                    ],
                    "data_source": datasets.Value("string"),
                    "data_split": datasets.Value("string")
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

    def _get_radgraph_entity(self, entity_id, entity_data, labeler):
        """Build radgraph entity from source entity JSON.
        
        Parameters
        ----------
        entity_id : string
            entity identifier from source data
        entity_data: dict
            entity record consisting of entity tokens, label, start index and end index
        labeler: string
            labeler identifier from source data
        
        Returns
        -------
        dict
            entity information
        """
        return {
                "labeler" : labeler,
                "entity_id": entity_id,
                "tokens": entity_data["tokens"],
                "label": entity_data["label"],
                "start_ix": entity_data["start_ix"],
                "end_ix": entity_data["end_ix"] 
                }

    def _get_radgraph_relations(self, relations_data, uid):
        """Build entity relations from source entity relations JSON.

        Parameters
        ----------
        relations_data: list
            list of relation records where each record is also a list, where the first element is the relation type, and the second element is the entity ID it refers to
        uid: int
            unique identifier
        
        Returns
        -------
        int
            unique identifier
        dict
            relations information
        """
        relations = []
        for relation_list in relations_data:
            relation = {
                "relation_id": str(uid),
                "type": relation_list[0],
                "arg": relation_list[1]
            }
            relations.append(relation)
            uid +=1
        return(uid, relations)

    def _parse_train_dev_data(self, data, chart_id, uid):
        """Parse train or dev JSON, return example"""
        example = {}
        entities = []
        chart_data = data[chart_id]
        example = { 
                "report_id": chart_id,
                "text" : chart_data["text"],
                "data_source" : chart_data["data_source"],
                "data_split": chart_data["data_split"]
        }
        for entity_id in chart_data["entities"]:
            entity_data = chart_data["entities"][entity_id]
            entity = self._get_radgraph_entity(entity_id, entity_data, "")
            uid, relations = self._get_radgraph_relations(entity_data["relations"], uid)
            
            entity["relations"] = relations
            entities.append(entity)

        example["entities"] = entities

        return(uid, example)

    def _parse_test_data(self, data, chart_id, uid):
        """Parse test JSON, return example"""
        example = {}
        entities = []
        chart_data = data[chart_id]
        
        example = { 
                "report_id": chart_id,
                "text" : chart_data["text"],
                "data_source" : chart_data["data_source"],
                "data_split": chart_data["data_split"],
        }

        for entity_id in chart_data["labeler_1"]["entities"]:
            entity_data = chart_data["labeler_1"]["entities"][entity_id]
            entity = self._get_radgraph_entity(entity_id, entity_data, "labeler_1")
            uid, relations = self._get_radgraph_relations(entity_data["relations"], uid)

            entity["relations"] = relations
            entities.append(entity)

        for entity_id in chart_data["labeler_2"]["entities"]:
            entity_data = chart_data["labeler_2"]["entities"][entity_id]

            entity = self._get_radgraph_entity(entity_id, entity_data, "labeler_2")
            uid, relations = self._get_radgraph_relations(entity_data["relations"], uid)
            
            entity["relations"] = relations
            entities.append(entity)

        example["entities"] = entities

        return(uid, example)

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        if self.config.schema == "source":
            with open(filepath) as json_file:
                data = json.load(json_file)
                uid = 0
                if "train" in filepath or "dev" in filepath:   
                    for chart_id in data:
                        uid, example = self._parse_train_dev_data(data, chart_id, uid)
                        yield uid, example
                        uid +=1
                elif "test" in filepath:
                    
                    for chart_id in data:
                        uid, example = self._parse_test_data(data, chart_id, uid)
                        yield uid, example
                        uid +=1

        # elif self.config.schema == "bigbio_kb":
        #     # TODO: yield (key, example) tuples in the bigbio schema
        #     for key, example in thing:
        #         yield key, example
