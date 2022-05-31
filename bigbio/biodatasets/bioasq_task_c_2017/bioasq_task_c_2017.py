# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
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

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks
from bigbio.utils.license import NLMLicense

_LOCAL = True
_CITATION = """\
@article{nentidis-etal-2017-results,
  author    = {Nentidis, Anastasios  and Bougiatiotis, Konstantinos  and Krithara, Anastasia  and
               Paliouras, Georgios  and Kakadiaris, Ioannis},
  title     = {Results of the fifth edition of the {B}io{ASQ} Challenge},
  journal   = {},
  volume    = {BioNLP 2017},
  year      = {2007},
  url       = {https://aclanthology.org/W17-2306},
  doi       = {10.18653/v1/W17-2306},
  biburl    = {},
  bibsource = {https://aclanthology.org/W17-2306}
}
"""

_DATASETNAME = "bioasq_task_c_2017"

_DESCRIPTION = """\
The training data set for this task contains annotated biomedical articles published in PubMed
and corresponding full text from PMC. By annotated is meant that GrantIDs and corresponding
Grant Agencies have been identified in the full text of articles.
"""

_HOMEPAGE = "http://participants-area.bioasq.org/general_information/Task5c/"

_LICENSE = NLMLicense

# Website contains all data, but login required
_URLS = {_DATASETNAME: "http://participants-area.bioasq.org/datasets/"}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

# this version doesn't have to be consistent with semantic versioning. Anything that is
# provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


@dataclass
class BioASQTaskC2017BigBioConfig(BigBioConfig):
    schema: str = "source"
    name: str = "bioasq_task_c_2017_source"
    version: datasets.Version = datasets.Version(_SOURCE_VERSION)
    description: str = "bioasq_task_c_2017 source schema"
    subset_id: str = "bioasq_task_c_2017"


class BioASQTaskC2017(datasets.GeneratorBasedBuilder):
    """
    BioASQ Task C Dataset for 2017
    """

    DEFAULT_CONFIG_NAME = "bioasq_task_c_2017_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BioASQTaskC2017BigBioConfig(
            name="bioasq_task_c_2017_source",
            version=SOURCE_VERSION,
            description="bioasq_task_c_2017 source schema",
            schema="source",
            subset_id="bioasq_task_c_2017",
        ),
        BioASQTaskC2017BigBioConfig(
            name="bioasq_task_c_2017_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_task_c_2017 BigBio schema",
            schema="bigbio_text",
            subset_id="bioasq_task_c_2017",
        ),
    ]

    BUILDER_CONFIG_CLASS = BioASQTaskC2017BigBioConfig

    def _info(self) -> datasets.DatasetInfo:

        # BioASQ Task C source schema
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "pmid": datasets.Value("string"),
                    "pmcid": datasets.Value("string"),
                    "grantList": [
                        {
                            "agency": datasets.Value("string"),
                        }
                    ],
                    "text": datasets.Value("string"),
                }
            )

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "bigbio_text":
            features = schemas.text.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "taskCTrainingData2017.json"),
                    "filespath": os.path.join(data_dir, "Train_Text"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "taskc_golden2.json"),
                    "filespath": os.path.join(data_dir, "Final_Text"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, filespath, split) -> (int, dict):

        with open(filepath) as f:
            task_data = json.load(f)

        if self.config.schema == "source":
            for article in task_data["articles"]:

                with open(filespath + "/" + article["pmcid"] + ".xml") as f:
                    text = f.read()
                pmid = article["pmid"]

                yield pmid, {
                    "text": text,  # articles[pmid],
                    "document_id": pmid,
                    "id": str(pmid),
                    "pmid": pmid,
                    "pmcid": article["pmcid"],
                    "grantList": [
                        {"agency": grant["agency"]} for grant in article["grantList"]
                    ],
                }

        elif self.config.schema == "bigbio_text":

            for article in task_data["articles"]:

                with open(filespath + "/" + article["pmcid"] + ".xml") as f:
                    xml_string = f.read()

                try:
                    article_body = ET.fromstring(xml_string).find("./article/body")
                except ET.ParseError:

                    # PubMed XML might not contain namespace which results in parse error, add manually
                    xml_string = xml_string.replace(
                        "</pmc-articleset>",
                        # xlink namespace
                        '<article xmlns:xlink="http://www.w3.org/1999/xlink"'  # mml namespace
                        ' xmlns:mml="http://www.w3.org/1998/Math/MathML"'
                        ' article-type="research-article">',
                    )
                    xml_string = xml_string + "</article></pmc-articleset>"
                    article_body = ET.fromstring(xml_string).find("./article/body")

                text = ET.tostring(article_body, encoding="utf8", method="text")

                yield article["pmid"], {
                    "text": text,
                    "id": str(article["pmid"]),
                    "document_id": article["pmid"],
                    "labels": [grant["agency"] for grant in article["grantList"]],
                }
