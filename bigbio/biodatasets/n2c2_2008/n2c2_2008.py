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
A dataset loader for the n2c2 2008 obesity and comorbidities dataset.

https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

The dataset consists of eight xml files,

* obesity_patient_records_training.xml
* obesity_patient_records_training2.xml
* obesity_standoff_annotations_training.xml
* obesity_standoff_annotations_training_addendum.xml
* obesity_standoff_annotations_training_addendum2.xml
* obesity_standoff_annotations_training_addendum3.xml
* obesity_patient_records_test.xml
* obesity_standoff_annotations_test.xml

containing patient records as well as textual and intuitive annotations.


The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_datset` via
the `data_dir` kwarg. This loader script will read the xml files
directly. For example, if the following directory structure exists
on the users local machine,


n2c2_2008
├── obesity_patient_records_training.xml
├── obesity_patient_records_training2.xml
├── obesity_standoff_annotations_training.xml
├── obesity_standoff_annotations_training_addendum.xml
├── obesity_standoff_annotations_training_addendum2.xml
├── obesity_standoff_annotations_training_addendum3.xml
├── obesity_patient_records_test.xml
├── obesity_standoff_annotations_test.xml


Data Access

from https://www.i2b2.org/NLP/DataSets/Main.php

"As always, you must register AND submit a DUA for access. If you previously
accessed the data sets here on i2b2.org, you will need to set a new password
for your account on the Data Portal, but your original DUA will be retained."


"""

import os
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_DATASETNAME = "n2c2_2008"

# https://academic.oup.com/jamia/article/16/4/561/766997
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = True
_CITATION = """\
@article{,
    author = {
        Uzuner, Ozlem
    },
    title     = {Recognizing Obesity and Comorbidities in Sparse Data},
    journal   = {Journal of the American Medical Informatics Association},
    volume    = {16},
    number    = {4},
    pages     = {561-570},
    year      = {2009},
    month     = {07},
    url       = {https://doi.org/10.1197/jamia.M3115},
    doi       = {10.1197/jamia.M3115},
    eprint    = {https://academic.oup.com/jamia/article-pdf/16/4/561/2302602/16-4-561.pdf}
}
"""

_DESCRIPTION = """\
The data for the n2c2 2008 obesity challenge consisted of discharge summaries from
the Partners HealthCare Research Patient Data Repository. These data were chosen 
from the discharge summaries of patients who were overweight or diabetic and had 
been hospitalized for obesity or diabetes sometime since 12/1/04. De-identification
was performed semi-automatically. All private health information was replaced with
synthetic identifiers.

The data for the challenge were annotated by two obesity experts from the 
Massachusetts General Hospital Weight Center. The experts were given a textual task, 
which asked them to classify each disease (see list of diseases above) as Present, 
Absent, Questionable, or Unmentioned based on explicitly documented information in 
the discharge summaries, e.g., the statement “the patient is obese”. The experts were 
also given an intuitive task, which asked them to classify each disease as Present, 
Absent, or Questionable by applying their intuition and judgment to information in 
the discharge summaries.
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = Licenses.DUA

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_CLASS_NAMES = ["present", "absent", "unmentioned", "questionable"]
_disease_names = [
    "Obesity",
    "Asthma",
    "CAD",
    "CHF",
    "Depression",
    "Diabetes",
    "Gallstones",
    "GERD",
    "Gout",
    "Hypercholesterolemia",
    "Hypertension",
    "Hypertriglyceridemia",
    "OA",
    "OSA",
    "PVD",
    "Venous Insufficiency",
]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


def _map_labels(doc, task):
    """
    Map obesity and comorbidity labels.
    :param doc: a document indexde by id
    :param task: textual or intuitive annotation task
    """
    lmap = {"Y": "present", "N": "absent", "U": "unmentioned", "Q": "questionable"}

    def _map_label(doc, task, label_name):
        if label_name in doc[task].keys():
            return lmap[doc[task][label_name]]
        else:
            return None

    if task in doc.keys():
        return {
            "Obesity": _map_label(doc, task, "Obesity"),
            "Asthma": _map_label(doc, task, "Asthma"),
            "CAD": _map_label(doc, task, "CAD"),
            "CHF": _map_label(doc, task, "CHF"),
            "Depression": _map_label(doc, task, "Depression"),
            "Diabetes": _map_label(doc, task, "Diabetes"),
            "Gallstones": _map_label(doc, task, "Gallstones"),
            "GERD": _map_label(doc, task, "GERD"),
            "Gout": _map_label(doc, task, "Gout"),
            "Hypercholesterolemia": _map_label(doc, task, "Hypercholesterolemia"),
            "Hypertension": _map_label(doc, task, "Hypertension"),
            "Hypertriglyceridemia": _map_label(doc, task, "Hypertriglyceridemia"),
            "OA": _map_label(doc, task, "OA"),
            "OSA": _map_label(doc, task, "OSA"),
            "PVD": _map_label(doc, task, "PVD"),
            "Venous Insufficiency": _map_label(doc, task, "Venous Insufficiency"),
        }
    else:
        return {task: None}


def _read_xml(partition, data_dir):
    """
    Load the data split.
    :param partition: train/test
    :param data_dir: train and test data directory
    """
    documents = {}
    all_diseases = set()
    notes = tuple()
    if partition == "train":
        with open(data_dir / "obesity_patient_records_training.xml") as t1, open(
            data_dir / "obesity_patient_records_training2.xml"
        ) as t2:
            notes1 = t1.read().strip()
            notes2 = t2.read().strip()
        notes = (notes1, notes2)
    elif partition == "test":
        with open(data_dir / "obesity_patient_records_test.xml") as t1:
            notes1 = t1.read().strip()
        notes = (notes1,)

    for file in notes:
        root = et.fromstring(file)
        root = root.findall("./docs")[0]
        for document in root.findall("./doc"):
            assert document.attrib["id"] not in documents
            documents[document.attrib["id"]] = {}
            documents[document.attrib["id"]]["text"] = document.findall("./text")[
                0
            ].text

    annotation_files = tuple()
    if partition == "train":
        with open(data_dir / "obesity_standoff_annotations_training.xml") as t1, open(
            data_dir / "obesity_standoff_annotations_training_addendum.xml"
        ) as t2, open(
            data_dir / "obesity_standoff_annotations_training_addendum2.xml"
        ) as t3, open(
            data_dir / "obesity_standoff_annotations_training_addendum3.xml"
        ) as t4:
            train1 = t1.read().strip()
            train2 = t2.read().strip()
            train3 = t3.read().strip()
            train4 = t4.read().strip()
        annotation_files = (train1, train2, train3, train4)
    elif partition == "test":
        with open(data_dir / "obesity_standoff_annotations_test.xml") as t1:
            test1 = t1.read().strip()
        annotation_files = (test1,)

    for file in annotation_files:
        root = et.fromstring(file)
        for diseases_annotation in root.findall("./diseases"):

            annotation_source = diseases_annotation.attrib["source"]
            assert isinstance(annotation_source, str)
            for disease in diseases_annotation.findall("./disease"):
                disease_name = disease.attrib["name"]
                all_diseases.add(disease_name)
                for annotation in disease.findall("./doc"):
                    doc_id = annotation.attrib["id"]
                    if not annotation_source in documents[doc_id]:
                        documents[doc_id][annotation_source] = {}
                    assert doc_id in documents
                    judgment = annotation.attrib["judgment"]
                    documents[doc_id][annotation_source][disease_name] = judgment
    return [
        {
            "document_id": str(id),
            "text": documents[id]["text"],
            "textual": _map_labels(documents[id], "textual"),
            "intuitive": _map_labels(documents[id], "intuitive"),
        }
        for id in documents
    ]


class N2C22008ObesityDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2008 obesity and comorbidities recognition task"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="n2c2_2008_source",
            version=SOURCE_VERSION,
            description="n2c2_2008 source schema",
            schema="source",
            subset_id="n2c2_2008",
        ),
        BigBioConfig(
            name="n2c2_2008_bigbio_text",
            version=BIGBIO_VERSION,
            description="n2c2_2008 BigBio schema",
            schema="bigbio_text",
            subset_id="n2c2_2008",
        ),
    ]

    DEFAULT_CONFIG_NAME = "n2c2_2008_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "labels": [
                        {
                            "annotation": datasets.ClassLabel(
                                names=["textual", "intuitive"]
                            ),
                            "disease_name": datasets.ClassLabel(names=_disease_names),
                            "label": datasets.ClassLabel(names=_CLASS_NAMES),
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
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
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "test",
                },
            ),
        ]

    @staticmethod
    def _get_source_sample(sample):
        textual_labels = [
            ("textual", disease_name, sample["textual"][disease_name])
            for disease_name in sample["textual"].keys()
            if sample["textual"][disease_name]
        ]
        intuitive_labels = [
            ("intuitive", disease_name, sample["intuitive"][disease_name])
            for disease_name in sample["intuitive"].keys()
            if sample["intuitive"][disease_name]
        ]

        return {
            "document_id": sample["document_id"],
            "text": sample["text"],
            "labels": [
                {
                    "annotation": label[0],
                    "disease_name": label[1],
                    "label": label[2],
                }
                for label in textual_labels + intuitive_labels
            ],
        }

    @staticmethod
    def _get_bigbio_sample(sample_id, sample):
        textual_labels = [
            ("textual", disease_name, sample["textual"][disease_name])
            for disease_name in sample["textual"].keys()
            if sample["textual"][disease_name]
        ]
        intuitive_labels = [
            ("intuitive", disease_name, sample["intuitive"][disease_name])
            for disease_name in sample["intuitive"].keys()
            if sample["intuitive"][disease_name]
        ]

        return {
            "id": str(sample_id),
            "document_id": sample["document_id"],
            "text": sample["text"],
            "labels": [
                {
                    "annotation": label[0],
                    "disease_name": label[1],
                    "label": label[2],
                }
                for label in textual_labels + intuitive_labels
            ],
        }

    def _generate_examples(self, data_dir, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        data_dir = Path(data_dir).resolve()
        if split == "train":
            _id = 0
            samples = _read_xml(split, data_dir)
            for sample in samples:
                if self.config.schema == "source":
                    yield _id, self._get_source_sample(sample)

                elif self.config.schema == "bigbio_text":
                    yield _id, self._get_bigbio_sample(_id, sample)
                _id += 1

        elif split == "test":
            _id = 0
            samples = _read_xml(split, data_dir)
            for sample in samples:
                if self.config.schema == "source":
                    yield _id, self._get_source_sample(sample)

                elif self.config.schema == "bigbio_text":
                    yield _id, self._get_bigbio_sample(_id, sample)
                _id += 1
