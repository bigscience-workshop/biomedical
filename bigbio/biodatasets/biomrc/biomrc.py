# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the
previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the
new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating
that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is
also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new
BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or
surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different
sizes, also releasing our code, and providing a leaderboard.
"""

import itertools as it
import json

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{pappas-etal-2020-biomrc,
    title = "{B}io{MRC}: A Dataset for Biomedical Machine Reading Comprehension",
    author = "Pappas, Dimitris  and
      Stavropoulos, Petros  and
      Androutsopoulos, Ion  and
      McDonald, Ryan",
    booktitle = "Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bionlp-1.15",
    pages = "140--149",
}
"""

_DATASETNAME = "biomrc"

_DESCRIPTION = """\
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the
previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the
new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating
that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is
also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new
BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or
surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different
sizes, also releasing our code, and providing a leaderboard.
"""

_HOMEPAGE = "https://github.com/PetrosStav/BioMRC_code"

_LICENSE = "Unknown"

_URLS = {
    "large": {
        "A": {
            "train": "https://archive.org/download/biomrc_dataset/biomrc_large/dataset_train.json.gz",
            "val": "https://archive.org/download/biomrc_dataset/biomrc_large/dataset_val.json.gz",
            "test": "https://archive.org/download/biomrc_dataset/biomrc_large/dataset_test.json.gz",
        },
        "B": {
            "train": "https://archive.org/download/biomrc_dataset/biomrc_large/dataset_train_B.json.gz",
            "val": "https://archive.org/download/biomrc_dataset/biomrc_large/dataset_val_B.json.gz",
            "test": "https://archive.org/download/biomrc_dataset/biomrc_large/dataset_test_B.json.gz",
        },
    },
    "small": {
        "A": {
            "train": "https://archive.org/download/biomrc_dataset/biomrc_small/dataset_train_small.json.gz",
            "val": "https://archive.org/download/biomrc_dataset/biomrc_small/dataset_val_small.json.gz",
            "test": "https://archive.org/download/biomrc_dataset/biomrc_small/dataset_test_small.json.gz",
        },
        "B": {
            "train": "https://archive.org/download/biomrc_dataset/biomrc_small/dataset_train_small_B.json.gz",
            "val": "https://archive.org/download/biomrc_dataset/biomrc_small/dataset_val_small_B.json.gz",
            "test": "https://archive.org/download/biomrc_dataset/biomrc_small/dataset_test_small_B.json.gz",
        },
    },
    "tiny": {
        "A": {"test": "https://archive.org/download/biomrc_dataset/biomrc_tiny/dataset_tiny.json.gz"},
        "B": {"test": "https://archive.org/download/biomrc_dataset/biomrc_tiny/dataset_tiny_B.json.gz"},
    },
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class BiomrcDataset(datasets.GeneratorBasedBuilder):
    """BioMRC: A Dataset for Biomedical Machine Reading Comprehension"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    for biomrc_setting in ["A", "B"]:
        for biomrc_version in ["large", "small", "tiny"]:
            subset_id = f"biomrc_{biomrc_version}_{biomrc_setting}"
            BUILDER_CONFIGS.append(
                BigBioConfig(
                    name=f"{subset_id}_source",
                    version=SOURCE_VERSION,
                    description=f"BioMRC Version {biomrc_version} Setting {biomrc_setting} source schema",
                    schema="source",
                    subset_id=subset_id,
                )
            )
            BUILDER_CONFIGS.append(
                BigBioConfig(
                    name=f"{subset_id}_bigbio_qa",
                    version=BIGBIO_VERSION,
                    description=f"BioMRC Version {biomrc_version} Setting {biomrc_setting} BigBio schema",
                    schema="bigbio_qa",
                    subset_id=subset_id,
                )
            )

    DEFAULT_CONFIG_NAME = "biomrc_large_B_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "abstract": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "entities_list": datasets.features.Sequence(
                        {
                            "pseudoidentifier": datasets.Value("string"),
                            "identifier": datasets.Value("string"),
                            "synonyms": datasets.Value("string"),
                        }
                    ),
                    "answer": {
                        "pseudoidentifier": datasets.Value("string"),
                        "identifier": datasets.Value("string"),
                        "synonyms": datasets.Value("string"),
                    },
                }
            )
        elif self.config.schema == "bigbio_qa":
            features = schemas.qa_features
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        _, version, setting = self.config.subset_id.split("_")
        downloaded_files = dl_manager.download_and_extract(_URLS[version][setting])

        if version == "tiny":
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["test"]}),
            ]
        else:
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["val"]}
                ),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
            ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as fp:
                biomrc = json.load(fp)
                for _id, (ab, ti, el, an) in enumerate(
                    zip(biomrc["abstracts"], biomrc["titles"], biomrc["entities_list"], biomrc["answers"])
                ):
                    el = [self._parse_dict_from_entity(entity) for entity in el]
                    an = self._parse_dict_from_entity(an)
                    yield _id, {"abstract": ab, "title": ti, "entities_list": el, "answer": an}
        elif self.config.schema == "bigbio_qa":
            with open(filepath, encoding="utf-8") as fp:
                uid = it.count(0)
                biomrc = json.load(fp)
                for _id, (ab, ti, el, an) in enumerate(
                    zip(biomrc["abstracts"], biomrc["titles"], biomrc["entities_list"], biomrc["answers"])
                ):
                    # remove info such as code, label, synonyms from answer and choices
                    # f.e. @entity1 :: ('9606', 'Species') :: ['patients', 'patient']"
                    example = {
                        "id": next(uid),
                        "question_id": next(uid),
                        "document_id": next(uid),
                        "question": ti,
                        "type": "multiple_choice",
                        "choices": [x.split(" :: ")[0] for x in el],
                        "context": ab,
                        "answer": [an.split(" :: ")[0]],
                    }
                    yield _id, example

    def _parse_dict_from_entity(self, entity):
        if "::" in entity:
            pseudoidentifier, identifier, synonyms = entity.split(" :: ")
            return {"pseudoidentifier": pseudoidentifier, "identifier": identifier, "synonyms": synonyms}
        else:
            return {"pseudoidentifier": entity, "identifier": "", "synonyms": ""}
