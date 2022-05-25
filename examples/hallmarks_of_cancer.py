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
import glob
import os

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@article{DBLP:journals/bioinformatics/BakerSGAHSK16,
  author    = {Simon Baker and
               Ilona Silins and
               Yufan Guo and
               Imran Ali and
               Johan H{\"{o}}gberg and
               Ulla Stenius and
               Anna Korhonen},
  title     = {Automatic semantic classification of scientific literature
               according to the hallmarks of cancer},
  journal   = {Bioinform.},
  volume    = {32},
  number    = {3},
  pages     = {432--440},
  year      = {2016},
  url       = {https://doi.org/10.1093/bioinformatics/btv585},
  doi       = {10.1093/bioinformatics/btv585},
  timestamp = {Thu, 14 Oct 2021 08:57:44 +0200},
  biburl    = {https://dblp.org/rec/journals/bioinformatics/BakerSGAHSK16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
The Hallmarks of Cancer (HOC) Corpus consists of 1852 PubMed publication
abstracts manually annotated by experts according to a taxonomy. The taxonomy
consists of 37 classes in a hierarchy. Zero or more class labels are assigned
to each sentence in the corpus. The labels are found under the "labels"
directory, while the tokenized text can be found under "text" directory.
The filenames are the corresponding PubMed IDs (PMID).
"""

_HOMEPAGE = "https://github.com/sb895/Hallmarks-of-Cancer"

_LICENSE = "GNU General Public License v3.0"

_URLs = {
    "source": "https://github.com/sb895/Hallmarks-of-Cancer/archive/refs/heads/master.zip",
    "bigbio_text": "https://github.com/sb895/Hallmarks-of-Cancer/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CLASS_NAMES = [
    "Activating invasion and metastasis",
    "Avoiding immune destruction",
    "Cellular energetics",
    "Enabling replicative immortality",
    "Evading growth suppressors",
    "Genomic instability and mutation",
    "Inducing angiogenesis",
    "Resisting cell death",
    "NULL",
    "Sustaining proliferative signaling",
    "Tumor promoting inflammation",
]


class HallmarksOfCancerDataset(datasets.GeneratorBasedBuilder):
    """Hallmarks Of Cancer Dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="hallmarks_of_cancer_source",
            version=SOURCE_VERSION,
            description="Hallmarks of Cancer source schema",
            schema="source",
            subset_id="hallmarks_of_cancer",
        ),
        BigBioConfig(
            name="hallmarks_of_cancer_bigbio_text",
            version=BIGBIO_VERSION,
            description="Hallmarks of Cancer Biomedical schema",
            schema="bigbio_text",
            subset_id="hallmarks_of_cancer",
        ),
    ]
    DEFAULT_CONFIG_NAME = "hallmarks_of_cancer_source"

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Sequence(datasets.ClassLabel(names=_CLASS_NAMES)),
                }
            )

        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split):

        path_name = list(filepath.values())[0] + "/*"
        texts = glob.glob(path_name + "/text/*")
        labels = glob.glob(path_name + "/labels/")
        uid = 1

        for idx, tf_name in enumerate(texts):
            filenname = os.path.basename(tf_name)
            with open(tf_name, encoding="utf-8") as f:
                lines = f.readlines()
                text_body = "".join([j.strip() for j in lines])

            label_file_name = labels[0] + "/" + filenname
            with open(label_file_name, encoding="utf-8") as f:
                lines = f.readlines()
                label_body = "".join([j.strip() for j in lines])
                label_body = [i.strip() for i in label_body.split("<")]
                label_body = sum([k.split("AND") for k in label_body if len(k) > 1], [])
                label_body = [i.split("--")[0].strip() for i in label_body]

            if self.config.schema == "source":
                yield idx, {
                    "document_id": filenname.split(".")[0],
                    "text": text_body,
                    "label": label_body,
                }
            elif self.config.schema == "bigbio_text":

                yield idx, {
                    "id": uid,
                    "document_id": filenname.split(".")[0],
                    "text": text_body,
                    "labels": [label_body],
                }

                uid += 1
