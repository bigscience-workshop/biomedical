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
A dataset loader for the n2c2 2018 cohort selection dataset.

The dataset consists of three archive files,
├── train.zip - 202 records
└── n2c2-t1_gold_standard_test_data.zip - 86 records

The individual data files (inside the zip and tar archives) come in
xml files that contains text as well as labels.


The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_dataset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files).

Data Access from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
"""

import os
import zipfile
from collections import defaultdict
from typing import List

import datasets
from lxml import etree

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks
from bigbio.utils.license import Licenses

_LOCAL = True
_CITATION = """\
@article{DBLP:journals/jamia/StubbsFSHU19,
  author    = {
                Amber Stubbs and
                Michele Filannino and
                Ergin Soysal and
                Samuel Henry and
                Ozlem Uzuner
               },
  title     = {Cohort selection for clinical trials: n2c2 2018 shared task track 1},
  journal   = {J. Am. Medical Informatics Assoc.},
  volume    = {26},
  number    = {11},
  pages     = {1163--1171},
  year      = {2019},
  url       = {https://doi.org/10.1093/jamia/ocz163},
  doi       = {10.1093/jamia/ocz163},
  timestamp = {Mon, 15 Jun 2020 16:56:11 +0200},
  biburl    = {https://dblp.org/rec/journals/jamia/StubbsFSHU19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "n2c2_2018_track1"

_DESCRIPTION = """\
Track 1 of the 2018 National NLP Clinical Challenges shared tasks focused
on identifying which patients in a corpus of longitudinal medical records
meet and do not meet identified selection criteria.

This shared task aimed to determine whether NLP systems could be trained to identify if patients met or did not meet
a set of selection criteria taken from real clinical trials. The selected criteria required measurement detection (
“Any HbA1c value between 6.5 and 9.5%”), inference (“Use of aspirin to prevent myocardial infarction”),
temporal reasoning (“Diagnosis of ketoacidosis in the past year”), and expert judgment to assess (“Major
diabetes-related complication”). For the corpus, we used the dataset of American English, longitudinal clinical
narratives from the 2014 i2b2/UTHealth shared task 4.

The final selected 13 selection criteria are as follows:
1. DRUG-ABUSE: Drug abuse, current or past
2. ALCOHOL-ABUSE: Current alcohol use over weekly recommended limits
3. ENGLISH: Patient must speak English
4. MAKES-DECISIONS: Patient must make their own medical decisions
5. ABDOMINAL: History of intra-abdominal surgery, small or large intestine
resection, or small bowel obstruction.
6. MAJOR-DIABETES: Major diabetes-related complication. For the purposes of
this annotation, we define “major complication” (as opposed to “minor complication”)
as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes:
    a. Amputation
    b. Kidney damage
    c. Skin conditions
    d. Retinopathy
    e. nephropathy
    f. neuropathy
7. ADVANCED-CAD: Advanced cardiovascular disease (CAD).
For the purposes of this annotation, we define “advanced” as having 2 or more of the following:
    a. Taking 2 or more medications to treat CAD
    b. History of myocardial infarction (MI)
    c. Currently experiencing angina
    d. Ischemia, past or present
8. MI-6MOS: MI in the past 6 months
9. KETO-1YR: Diagnosis of ketoacidosis in the past year
10. DIETSUPP-2MOS: Taken a dietary supplement (excluding vitamin D) in the past 2 months
11. ASP-FOR-MI: Use of aspirin to prevent MI
12. HBA1C: Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%
13. CREATININE: Serum creatinine > upper limit of normal

The training consists of 202 patient records with document-level annotations, 10 records
with textual spans indicating annotator’s evidence for their annotations while test set contains 86.

Note: 
* The inter-annotator average agreement is 84.9%
* Whereabouts of 10 records with textual spans indicating annotator’s evidence are unknown. 
However, author did a simple script based validation to check if any of the tags contained any text 
in any of the training set and they do not, which confirms that atleast train and test do not
 have any evidence tagged alongside corresponding tags.
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = Licenses.EXTERNAL_DUA

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

# Constants
SOURCE = "source"
BIGBIO_TEXT = "bigbio_text"


def _read_zip(file_path):
    samples = defaultdict(dict)
    with zipfile.ZipFile(file_path) as zf:
        for info in zf.infolist():

            base, filename = os.path.split(info.filename)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            sample_id = filename.split(".")[0]

            if ext == "xml" and not filename.startswith("."):
                content = zf.read(info).decode("utf-8").encode()
                root = etree.XML(content)
                text, tags = root.getchildren()
                samples[sample_id]["txt"] = text.text
                samples[sample_id]["tags"] = {}
                for child in tags:
                    samples[sample_id]["tags"][child.tag] = child.get("met")

    return samples


class N2C22018CohortSelectionDataset(datasets.GeneratorBasedBuilder):
    """i2b2 2018 track 1 cohort selection task"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    _SOURCE_CONFIG_NAME = _DATASETNAME + "_" + SOURCE
    _BIGBIO_CONFIG_NAME = _DATASETNAME + "_" + BIGBIO_TEXT

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=_SOURCE_CONFIG_NAME,
            version=SOURCE_VERSION,
            description=_DATASETNAME + " source schema",
            schema=SOURCE,
            subset_id=_DATASETNAME,
        ),
        BigBioConfig(
            name=_BIGBIO_CONFIG_NAME,
            version=BIGBIO_VERSION,
            description=_DATASETNAME + " BigBio schema",
            schema=BIGBIO_TEXT,
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = _SOURCE_CONFIG_NAME
    LABEL_CLASS_NAMES = [
        "ABDOMINAL",
        "ADVANCED-CAD",
        "ALCOHOL-ABUSE",
        "ASP-FOR-MI",
        "CREATININE",
        "DIETSUPP-2MOS",
        "DRUG-ABUSE",
        "ENGLISH",
        "HBA1C",
        "KETO-1YR",
        "MAJOR-DIABETES",
        "MAKES-DECISIONS",
        "MI-6MOS",
    ]

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == SOURCE:
            labels = {
                key: datasets.ClassLabel(names=["met", "not met"])
                for key in self.LABEL_CLASS_NAMES
            }
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tags": labels,
                }
            )

        elif self.config.schema == BIGBIO_TEXT:
            features = schemas.text.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        if self.config.data_dir is None or self.config.name is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir and name kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "train.zip"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file_path": os.path.join(
                        data_dir, "n2c2-t1_gold_standard_test_data.zip"
                    ),
                },
            ),
        ]

    @staticmethod
    def _get_source_sample(sample_id, sample):
        return {
            "id": sample_id,
            "document_id": sample_id,
            "text": sample.get("txt", ""),
            "tags": sample.get("tags", {}),
        }

    @staticmethod
    def _get_bigbio_sample(sample_id, sample) -> dict:

        tags = sample.get("tags", None)
        if tags:
            labels = [name for name, met_status in tags.items() if met_status == "met"]
        else:
            labels = []

        return {
            "id": sample_id,
            "document_id": sample_id,
            "text": sample.get("txt", ""),
            "labels": labels,
        }

    def _generate_examples(self, file_path) -> (int, dict):
        samples = _read_zip(file_path)

        _id = 0
        for sample_id, sample in samples.items():

            if self.config.name == self._SOURCE_CONFIG_NAME:
                yield _id, self._get_source_sample(sample_id, sample)
            elif self.config.name == self._BIGBIO_CONFIG_NAME:
                yield _id, self._get_bigbio_sample(sample_id, sample)

            _id += 1
