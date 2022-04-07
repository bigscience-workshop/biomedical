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
import os
import json
import glob
import datasets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple
from xml.etree import ElementTree as ET

import utils.parsing as parsing
import utils.schemas as schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@inproceedings{MEDIQA2019,
  author    = {Asma {Ben Abacha} and Chaitanya Shivade and Dina Demner{-}Fushman},
  title     = {Overview of the MEDIQA 2019 Shared Task on Textual Inference, Question Entailment and Question Answering}, 
  booktitle = {ACL-BioNLP 2019},  
  year      = {2019}
}
"""

_DATASETNAME = "mediqa_rqe"

_DESCRIPTION = """\
The MEDIQA challenge is an ACL-BioNLP 2019 shared task aiming to attract further research efforts in Natural Language Inference (NLI), Recognizing Question Entailment (RQE), and their applications in medical Question Answering (QA).  
Mailing List: https://groups.google.com/forum/#!forum/bionlp-mediqa 

The objective of the RQE task is to identify entailment between two questions in the context of QA. We use the following definition of question entailment: “a question A entails a question B if every answer to B is also a complete or partial answer to A” [1]
    [1] A. Ben Abacha & D. Demner-Fushman. “Recognizing Question Entailment for Medical Question Answering”. AMIA 2016.
"""

_HOMEPAGE = "https://sites.google.com/view/mediqa2019"
_LICENSE = "-"

_URLS = {
    _DATASETNAME: "https://github.com/abachaa/MEDIQA2019/archive/refs/heads/master.zip"
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

class MediqaRQEDataset(datasets.GeneratorBasedBuilder):
    """MediqaRQE Dataset"""
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        # Source Schema
        BigBioConfig(
            name="mediqa_rqe_source",
            version=SOURCE_VERSION,
            description="MediqaRQE source schema",
            schema="source",
            subset_id="mediqa_rqe_source",
        ),
        # BigBio Schema
        BigBioConfig(
            name="mediqa_rqe_bigbio_te",
            version=BIGBIO_VERSION,
            description="MediqaRQE BigBio schema",
            schema="bigbio_te",
            subset_id="mediqa_rqe_bigbio_te",
        )
    ]
    
    DEFAULT_CONFIG_NAME = "mediqa_rqe_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {                    
                    "pid": datasets.Value("string"),
                    "value": datasets.Value("string"),
                    "chq": datasets.Value("string"),
                    "faq": datasets.Value("string")
                }
            )
        elif self.config.schema == "bigbio_te":
            features = schemas.entailment_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir / "MEDIQA2019-master/MEDIQA_Task2_RQE/MEDIQA2019-Task2-RQE-TrainingSet-AMIA2016.xml"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir / "MEDIQA2019-master/MEDIQA_Task2_RQE/MEDIQA2019-Task2-RQE-ValidationSet-AMIA2016.xml"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir / "MEDIQA2019-master/MEDIQA_Task2_RQE/MEDIQA2019-Task2-RQE-TestSet-wLabels.xml"
                }
            )
        ]            

    def _generate_examples(self, filepath: Path) -> Iterator[Tuple[str, Dict]]:
        dom = ET.parse(filepath).getroot()
        for row in dom.iterfind('pair'):
            pid = row.attrib['pid']
            value = row.attrib['value']
            chq = row.find('chq').text
            faq = row.find('faq').text
            
            if self.config.schema == "source":
                yield pid, {                    
                    "pid": pid,
                    "value": value,
                    "chq": chq,
                    "faq": faq
                }
            elif self.config.schema == "bigbio_te":
                yield pid, {
                    "id": pid,
                    "premise": chq,
                    "hypothesis": faq,
                    "label": value,
                }

