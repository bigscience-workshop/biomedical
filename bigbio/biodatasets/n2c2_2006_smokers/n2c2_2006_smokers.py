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
A dataset loader for the n2c2 2006 smoking status dataset.

https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

The dataset consists of two archive files,

* smokers_surrogate_train_all_version2.zip
* smokers_surrogate_test_all_groundtruth_version2.zip

The individual data files (inside the zip archives) come in just 1 type:

* xml (*.xml files): contains the id and text of the patient records,
and corresponding smoking status labels


The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_datset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files). For example, if the following directory structure exists
on the users local machine,


n2c2_2006
├── smokers_surrogate_train_all_version2.zip
├── smokers_surrogate_test_all_groundtruth_version2.zip


Data Access

from https://www.i2b2.org/NLP/DataSets/Main.php

"As always, you must register AND submit a DUA for access. If you previously
accessed the data sets here on i2b2.org, you will need to set a new password
for your account on the Data Portal, but your original DUA will be retained."


"""

import os
import xml.etree.ElementTree as et
import zipfile
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_DATASETNAME = "n2c2_2006"

# https://academic.oup.com/jamia/article/15/1/14/779738
_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = True
_CITATION = """\
@article{,
    author = {
        Uzuner, Ozlem and
        Goldstein, Ira and
        Luo, Yuan and
        Kohane, Isaac
    },
    title     = {Identifying Patient Smoking Status from Medical Discharge Records},
    journal   = {Journal of the American Medical Informatics Association},
    volume    = {15},
    number    = {1},
    pages     = {14-24},
    year      = {2008},
    month     = {01},
    url       = {https://doi.org/10.1197/jamia.M2408},
    doi       = {10.1136/amiajnl-2011-000784},
    eprint    = {https://academic.oup.com/jamia/article-pdf/15/1/14/2339646/15-1-14.pdf}
}
"""

_DESCRIPTION = """\
The data for the n2c2 2006 smoking challenge consisted of discharge summaries
from Partners HealthCare, which were then de-identified, tokenized, broken into
sentences, converted into XML format, and separated into training and test sets.

Two pulmonologists annotated each record with the smoking status of patients based
strictly on the explicitly stated smoking-related facts in the records. These
annotations constitute the textual judgments of the annotators. The annotators
were asked to classify patient records into five possible smoking status categories:
a past smoker, a current smoker, a smoker, a non-smoker and an unknown. A total of
502 de-identified medical discharge records were used for the smoking challenge.
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "Data User Agreement"

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CLASS_NAMES = ["current smoker", "non-smoker", "past smoker", "smoker", "unknown"]


def _read_zip(file_path):
    _, filename = os.path.split(file_path)
    zipped = zipfile.ZipFile(file_path, "r")
    file = zipped.read(filename.split(".")[0] + ".xml")

    root = et.fromstring(file)
    ids = []
    notes = []
    labels = []
    documents = root.findall("./RECORD")
    for document in documents:
        ids.append(document.attrib["ID"])
        notes.append(document.findall("./TEXT")[0].text)
        labels.append(document.findall("./SMOKING")[0].attrib["STATUS"].lower())
    return [(id, note, label) for id, note, label in zip(ids, notes, labels)]


class N2C22006SmokingDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2006 smoking status identification task"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="n2c2_2006_smokers_source",
            version=SOURCE_VERSION,
            description="n2c2_2006_smokers source schema",
            schema="source",
            subset_id="n2c2_2006_smokers",
        ),
        BigBioConfig(
            name="n2c2_2006_smokers_bigbio_text",
            version=BIGBIO_VERSION,
            description="n2c2_2006_smokers BigBio schema",
            schema="bigbio_text",
            subset_id="n2c2_2006_smokers",
        ),
    ]

    DEFAULT_CONFIG_NAME = "n2c2_2006_smokers_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=_CLASS_NAMES),
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

    def _generate_examples(self, data_dir, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if split == "train":
            _id = 0
            path = os.path.join(data_dir, "smokers_surrogate_train_all_version2.zip")
            samples = _read_zip(path)
            for sample in samples:
                if self.config.schema == "source":
                    yield _id, {
                        "document_id": sample[0],
                        "text": sample[1],
                        "label": sample[-1],
                    }
                elif self.config.schema == "bigbio_text":
                    yield _id, {
                        "id": sample[0],
                        "document_id": sample[0],
                        "text": sample[1],
                        "labels": [sample[-1]],
                    }
                _id += 1

        elif split == "test":
            _id = 0
            path = os.path.join(
                data_dir, "smokers_surrogate_test_all_groundtruth_version2.zip"
            )
            samples = _read_zip(path)
            for sample in samples:
                if self.config.schema == "source":
                    yield _id, {
                        "document_id": sample[0],
                        "text": sample[1],
                        "label": sample[-1],
                    }
                elif self.config.schema == "bigbio_text":
                    yield _id, {
                        "id": sample[0],
                        "document_id": sample[0],
                        "text": sample[1],
                        "labels": [sample[-1]],
                    }
                _id += 1
