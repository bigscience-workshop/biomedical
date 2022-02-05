# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
A dataset loader for the n2c2 2011 coref dataset.

https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_datset` via
the `data_dir` kwarg.

```
ds = load_dataset("n2c2_2011_coref.py", name="original", data_dir="/path/to/dataset/files")
```

* Task_1C.zip
* Task_1C_Test_groundtruth.zip
* i2b2_Partners_Train_Release.tar.gz
* i2b2_Beth_Train_Release.tar.gz

The individual data files (inside the zip and tar archives)come in 4 types,

* chains (*.txt.chains files): chains (i.e. one or more) coreferent entities
* concepts (*.txt.con files): entities used as input to a coreference model
* docs (*.txt files): text of a patient record
* pairs (*.txt.pairs files): pairs of coreferent entities


TODO: Figure out canonical coreference schema
probably something like,
* text
* candidate entities (IDs, offsets, types, ...)
* coref sets (groups of entity IDs that are coreferent)


Data Access

from https://www.i2b2.org/NLP/DataSets/Main.php

"As always, you must register AND submit a DUA for access. If you previously
accessed the data sets here on i2b2.org, you will need to set a new password
for your account on the Data Portal, but your original DUA will be retained."


"""

from collections import defaultdict
import logging
import os
import tarfile
from typing import Iterable, Dict, List
import zipfile

import datasets


_DATASETNAME = "n2c2_2011_coref"

# https://academic.oup.com/jamia/article/19/5/786/716138
_CITATION = """\
@article{,
    author = {Uzuner, Ozlem and Bodnari, Andreea and Shen, Shuying and Forbush, Tyler and Pestian, John and South, Brett R},
    title = "{Evaluating the state of the art in coreference resolution for electronic medical records}",
    journal = {Journal of the American Medical Informatics Association},
    volume = {19},
    number = {5},
    pages = {786-791},
    year = {2012},
    month = {02},
    issn = {1067-5027},
    doi = {10.1136/amiajnl-2011-000784},
    url = {https://doi.org/10.1136/amiajnl-2011-000784},
    eprint = {https://academic.oup.com/jamia/article-pdf/19/5/786/17374287/19-5-786.pdf},
}
"""

_DESCRIPTION = """\
The i2b2/VA corpus contained de-identified discharge summaries from Beth Israel
Deaconess Medical Center, Partners Healthcare, and University of Pittsburgh Medical
Center (UPMC). In addition, UPMC contributed de-identified progress notes to the
i2b2/VA corpus. This dataset contains the records from Beth Israel and Partners.

The i2b2/VA corpus contained five concept categories: problem, person, pronoun,
test, and treatment. Each record in the i2b2/VA corpus was annotated by two
independent annotators for coreference pairs. Then the pairs were post-processed
in order to create coreference chains. These chains were presented to an adjudicator,
who resolved the disagreements between the original annotations, and added or deleted
annotations as necessary. The outputs of the adjudicators were then re-adjudicated, with
particular attention being paid to duplicates and enforcing consistency in the annotations.

"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "Data User Agreement"

_VERSION = "1.0.0"



class N2C22011CorefDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2011 coreference task"""

    VERSION = datasets.Version(_VERSION)

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="original",
            version=VERSION,
            description="original form of data",
        ),
        datasets.BuilderConfig(
            name="coref",
            version=VERSION,
            description="canonical coref form of data",
        )
    ]

    DEFAULT_CONFIG_NAME = _DATASETNAME

    def _info(self):

        if self.config.name == "original":
            features = datasets.Features(
                {
                    "sample_id": datasets.Value("string"),
                    "txt": datasets.Value("string"),
                    "con": datasets.Value("string"),
                    "pairs": datasets.Value("string"),
                    "chains": datasets.Value("string"),
                    "metadata": {
                        "txt_source": datasets.Value("string"),
                        "con_source": datasets.Value("string"),
                        "pairs_source": datasets.Value("string"),
                        "chains_source": datasets.Value("string"),
                    }
                }
            )

        elif self.config.name == "coref":
            features = datasets.Features(
                {
                    "passages": [
                        {
                            "document_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),

                        }
                    ]
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """
        self.config.data_dir and self.config.data_files can be made available by
        passing the `data_dir` and/or `data_files` kwargs to `load_dataset`.

        dataset = datasets.load_dataset(
            "n2c2_2011_coref.py",
            name="original",
            data_dir="path/to/n2c2_2011_coref/data"
        )

        """

        data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                },
            ),
        ]


    @staticmethod
    def _read_tar_gz(file_path, samples=None):
        if samples is None:
            samples = defaultdict(dict)
        with tarfile.open(file_path, "r:gz") as tf:
            for member in tf.getmembers():

                base, filename = os.path.split(member.name)
                _, ext = os.path.splitext(filename)
                ext = ext[1:] # get rid of dot
                sample_id = filename.split('.')[0]

                if ext in ["txt", "con", "pairs", "chains"]:
                    samples[sample_id][f"{ext}_source"] = os.path.basename(file_path) + "|" + member.name
                    with tf.extractfile(member) as fp:
                        content_bytes = fp.read()
                    content = content_bytes.decode("utf-8")
                    samples[sample_id][ext] = content

        return samples


    @staticmethod
    def _read_zip(file_path, samples=None):
        if samples is None:
            samples = defaultdict(dict)
        with zipfile.ZipFile(file_path) as zf:
            for info in zf.infolist():

                base, filename = os.path.split(info.filename)
                _, ext = os.path.splitext(filename)
                ext = ext[1:] # get rid of dot
                sample_id = filename.split('.')[0]

                if ext in ["txt", "con", "pairs", "chains"] and not filename.startswith("."):
                    samples[sample_id][f"{ext}_source"] = os.path.basename(file_path) + "|" + info.filename
                    content = zf.read(info).decode("utf-8")
                    samples[sample_id][ext] = content

        return samples


    @staticmethod
    def _get_original_sample(sample_id, sample):
        return {
            "sample_id": sample_id,
            "txt": sample.get("txt", ""),
            "con": sample.get("con", ""),
            "pairs": sample.get("pairs", ""),
            "chains": sample.get("chains", ""),
            "metadata": {
                "txt_source": sample.get("txt_source", ""),
                "con_source": sample.get("con_source", ""),
                "pairs_source": sample.get("pairs_source", ""),
                "chains_source": sample.get("chains_source", ""),
            }
        }


    @staticmethod
    def _get_coref_sample(sample_id, sample):
        return {
            "passages": [
                {
                    "document_id": sample_id,
                    "type": "discharge summary",
                    "text": sample.get("txt", ""),
                 }
            ]
        }



    def _generate_examples(self, split):
        """

        """
        if split=="train":
            _id = 0
            # These files have complete sample info
            # (so we get a fresh `samples` defaultdict from each)
            paths = [
                os.path.join(self.config.data_dir, "i2b2_Beth_Train_Release.tar.gz"),
                os.path.join(self.config.data_dir, "i2b2_Partners_Train_Release.tar.gz"),
            ]
            for path in paths:
                samples = self._read_tar_gz(path)
                for sample_id, sample in samples.items():
                    if self.config.name == "original":
                        yield _id, self._get_original_sample(sample_id, sample)
                    elif self.config.name == "coref":
                        yield _id, self._get_coref_sample(sample_id, sample)
                    _id += 1

        elif split == "test":
            _id = 0
            # Information from these files has to be combined to create a full sample
            # (so we pass the `samples` defaultdict back to the `_read_zip` method)
            paths = [
                os.path.join(self.config.data_dir, "Task_1C.zip"),
                os.path.join(self.config.data_dir, "Task_1C_Test_groundtruth.zip"),
            ]
            samples = defaultdict(dict)
            for path in paths:
                samples = self._read_zip(path, samples=samples)

            for sample_id, sample in samples.items():
                if self.config.name == "original":
                    yield _id, self._get_original_sample(sample_id, sample)
                elif self.config.name == "coref":
                    yield _id, self._get_coref_sample(sample_id, sample)
                _id += 1
