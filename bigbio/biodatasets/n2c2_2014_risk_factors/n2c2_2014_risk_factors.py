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
* training-RiskFactors-Gold-Set1.tar.gz
* training-RiskFactors-Gold-Set2.tar.gz
* testing-RiskFactors-Gold.tar.gz
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
├── training-RiskFactors-Gold-Set1.tar.gz
├── training-RiskFactors-Gold-Set2.tar.gz
├── testing-RiskFactors-Gold.tar.gz
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

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_LOCAL = True
_CITATION = """\
@article{
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

_DATASETNAME = "n2c2_2014_risk_factors"

_DESCRIPTION = """\
The 2014 i2b2/UTHealth Natural Language Processing (NLP) shared task featured two tracks.
The first of these was the de-identification track focused on identifying protected health
information (PHI) in longitudinal clinical narratives. The second track focused on the
identification of cardiac risk factors

TRACK 2: CARDIAC RISK FACTORS\n
This task consist on a set of medical documents that track the progression of heart disease
in diabetic patients. Multiple records are annotated for each patient, which will allow a general
timeline from the set. This project uses tags and attributes used to indicate the presence
and progression of disease (diabetes, heart disease), associated risk factors (hypertension,
hyperlipidemia, smoking status, obesity status, and family history), and the time they were present in
the patient's medical history.
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE =  Licenses.DUA
# _LICENSE = "DUA-C/NC"

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class N2C22014RiskFactorsDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2014"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="n2c2_2014_source",
            version=SOURCE_VERSION,
            description="n2c2_2014 source schema",
            schema="source",
            subset_id="n2c2_2014_risk_factors",
        ),
        BigBioConfig(
            name="n2c2_2014_bigbio_text",
            version=BIGBIO_VERSION,
            description="n2c2_2014 BigBio schema",
            schema="bigbio_text",
            subset_id="n2c2_2014_risk_factors",
        )
    ]

    DEFAULT_CONFIG_NAME = "n2c2_2014_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "cardiac_risk_factors": [
                        {
                            "id": datasets.Value("string"),
                            "risk_factor": datasets.Value("string"),
                            "indicator": datasets.Value("string"),
                            "time": datasets.Value("string"),
                            "status": datasets.Value("string"),
                            "type1": datasets.Value("string"),
                            "type2": datasets.Value("string"),
                            "comment": datasets.Value("string"),
                        }
                    ],
                },
            )

        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "file_names": [
                        ("training-RiskFactors-Gold-Set1.tar.gz", "track2"),
                        ("training-RiskFactors-Gold-Set2.tar.gz", "track2"),
                    ],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "file_names": [
                        ("testing-RiskFactors-Gold.tar.gz", "track2"),
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
                        document = self._read_task2_file(file_object=x["file_object"], file_name=x["file_name"])
                        document["id"] = next(uid)

        elif self.config.schema == "bigbio_text":
            uid = it.count(0)
            for fname, task in file_names:
                full_path = os.path.join(data_dir, fname)
                for x in self._read_tar_gz(full_path):
                    xml_flag = x["xml_flag"]
                    if xml_flag:
                        if task == "track2":
                            document = self._read_task2_file(file_object=x["file_object"], file_name=x["file_name"])
                            document["id"] = next(uid)

                            labels = []
                            risk_factors = document.pop("cardiac_risk_factors")
                            for label in risk_factors:
                                label_list = []
                                for key, value in label.items():
                                    if value != "" and key not in ["comment", "id"]:
                                        label_list.append(value)
                                labels.append("-".join(label_list))
                            document["labels"] = labels

                            yield document["id"], document
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
            yield {"file_object": file_object, "file_name": file_name, "xml_flag": xml_flag}

    def _read_task2_file(self, file_object, file_name):
        # Fix the file
        bad_tag = b"<?xml version='1.0' encoding='UTF-8'?>"
        old_xml = file_object.read()
        new_xml = old_xml.replace(bad_tag, bytes())
        xmldoc = et.fromstring(new_xml)

        # Find all tags
        entities = xmldoc.findall("TAGS")[0]
        text = xmldoc.findall("TEXT")[0].text
        risk_factors = []
        for entity in entities:
            risk_factor = {x: "" for x in ["indicator", "time", "status", "type1", "type2", "comment"]}
            for key, value in entity.attrib.items():
                risk_factor[key] = value

            risk_factor["risk_factor"] = entity.tag
            risk_factors.append(risk_factor)

        document = {"document_id": file_name, "text": text, "cardiac_risk_factors": risk_factors}
        return document
