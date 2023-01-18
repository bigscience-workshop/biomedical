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
A dataset loader for the n2c2 2014  Deidentification & Heart Disease.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
The dataset consists of 3  archive files,
* 2014_training-PHI-Gold-Set1.tar.gz
* training-PHI-Gold-Set2.tar.gz
* testing-PHI-Gold-fixed.tar.gz
Each tar.gz contain a set of .xml files. One .xml per clinical report.
The file names follow a consistent pattern with the first set of digits identifying the
patient and the last set of digits identifying the sequential record number
ie: XXX-YY.xml
where XXX is the patient number,  and YY is the record number.
Example: 320-03.xml
This is the third (03) record for patient 320
Each file has a root level xml node which will contain a
<TEXT> node that holds the medical annotation text and a <TAGS> node containing
annotations for the document text.
The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_datset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files). For example, if the following directory structure exists
on the users local machine,
n2c2_2014
├── 2014_training-PHI-Gold-Set1.tar.gz
├── training-PHI-Gold-Set2.tar.gz
├── testing-PHI-Gold-fixed.tar.gz
Data Access
from https://www.i2b2.org/NLP/DataSets/Main.php
"As always, you must register AND submit a DUA for access. If you previously
accessed the data sets here on i2b2.org, you will need to set a new password
for your account on the Data Portal, but your original DUA will be retained."
Made in collaboration with @JoaoRacedo
"""

import itertools as it
import os
import re
import tarfile
import xml.etree.ElementTree as et
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import kb_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ['English']
_PUBMED = False
_LOCAL = True
_CITATION = """\
@article{stubbs2015automated,
title = {Automated systems for the de-identification of longitudinal
clinical narratives: Overview of 2014 i2b2/UTHealth shared task Track 1},
journal = {Journal of Biomedical Informatics},
volume = {58},
pages = {S11-S19},
year = {2015},
issn = {1532-0464},
doi = {https://doi.org/10.1016/j.jbi.2015.06.007},
url = {https://www.sciencedirect.com/science/article/pii/S1532046415001173},
author = {Amber Stubbs and Christopher Kotfila and Özlem Uzuner}
}
"""

_DATASETNAME = "n2c2_2014_deid"
_DISPLAYNAME = "n2c2 2014 De-identification"

_DESCRIPTION = """\
The 2014 i2b2/UTHealth Natural Language Processing (NLP) shared task featured two tracks.
The first of these was the de-identification track focused on identifying protected health
information (PHI) in longitudinal clinical narratives.

TRACK 1: NER PHI\n
HIPAA requires that patient medical records have all identifying information removed in order to
protect patient privacy. There are 18 categories of Protected Health Information (PHI) identifiers of the
patient or of relatives, employers, or household members of the patient that must be removed in order
for a file to be considered de-identified.
In order to de-identify the records, each file has PHI marked up. All PHI has an
XML tag indicating its category and type, where applicable. For the purposes of this task,
the 18 HIPAA categories have been grouped into 6 main categories and 25 sub categories
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = 'Data User Agreement'

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class N2C22014DeidDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2014 Deidentification Challenge"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="n2c2_2014_source",
            version=SOURCE_VERSION,
            description="n2c2_2014 source schema",
            schema="source",
            subset_id="n2c2_2014_deid",
        ),
        BigBioConfig(
            name="n2c2_2014_bigbio_kb",
            version=BIGBIO_VERSION,
            description="n2c2_2014 BigBio schema",
            schema="bigbio_kb",
            subset_id="n2c2_2014_deid",
        ),
    ]

    DEFAULT_CONFIG_NAME = "n2c2_2014_deid_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "phi": [
                        {
                            "id": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "comment": datasets.Value("string"),
                        }
                    ],
                },
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
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
                    "data_dir": data_dir,
                    "file_names": [
                        ("2014_training-PHI-Gold-Set1.tar.gz", "track1"),
                        ("training-PHI-Gold-Set2.tar.gz", "track1"),
                    ],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "file_names": [
                        ("testing-PHI-Gold-fixed.tar.gz", "track1"),
                    ],
                },
            ),
        ]

    def _generate_examples(self, data_dir, file_names: List[Tuple]) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            uid = it.count(0)
            for fname, task in file_names:
                full_path = os.path.join(data_dir, fname)
                for x in self._read_tar_gz(full_path):
                    xml_flag = x["xml_flag"]
                    if xml_flag:
                        document = self._read_task1_file(
                            file_object=x["file_object"], file_name=x["file_name"]
                        )
                        document["id"] = next(uid)

        elif self.config.schema == "bigbio_kb":
            uid = it.count(0)
            for fname, task in file_names:
                full_path = os.path.join(data_dir, fname)
                for x in self._read_tar_gz(full_path):
                    xml_flag = x["xml_flag"]
                    if xml_flag:
                        document = self._read_task1_file(
                            file_object=x["file_object"], file_name=x["file_name"]
                        )
                        document["id"] = next(uid)
                        entity_list = document.pop("phi")
                        full_text = document.pop("text")
                        entities_ = []
                        for entity in entity_list:
                            entities_.append(
                                {
                                    "id": next(uid),
                                    "type": entity["type"],
                                    "text": entity["text"],
                                    "offsets": entity["offsets"],
                                    "normalized": entity["normalized"],
                                }
                            )
                        document["entities"] = entities_

                        document["passages"] = [
                            {
                                "id": next(uid),
                                "type": "full_text",
                                "text": [full_text],
                                "offsets": [[0, len(full_text)]],
                            },
                        ]

                        # additional fields required that can be empty
                        document["relations"] = []
                        document["events"] = []
                        document["coreferences"] = []
                        yield document["document_id"], document
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

    def _read_tar_gz(self, fpath: str) -> Dict:
        """
        Read .tar.gz file
        """
        # Open tar file
        tf = tarfile.open(fpath, "r:gz")

        for tf_member in tf.getmembers():
            file_object = tf.extractfile(tf_member)
            name = tf_member.name
            file_name = os.path.basename(name).split(".")[0]
            if re.search(r"\.xml", name) is not None:
                xml_flag = True
            else:
                xml_flag = False
            yield {
                "file_object": file_object,
                "file_name": file_name,
                "xml_flag": xml_flag,
            }

    def _read_task1_file(self, file_object, file_name):
        xmldoc = et.parse(file_object).getroot()
        entities = xmldoc.findall("TAGS")[0]
        text = xmldoc.findall("TEXT")[0].text
        phi = []
        for entity in entities:
            phi.append(
                {
                    "id": entity.attrib["id"],
                    "offsets": [[entity.attrib["start"], entity.attrib["end"]]],
                    "type": entity.attrib["TYPE"],
                    "text": [entity.attrib["text"]],
                    "comment": entity.attrib["comment"],
                    "normalized": [],
                }
            )

        document = {"document_id": file_name, "text": text, "phi": phi}
        return document
