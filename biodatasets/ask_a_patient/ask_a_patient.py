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

import os
import re
import datasets
import glob
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks


_DATASETNAME = "ask_a_patient"

_CITATION = """
@inproceedings{limsopatham-collier-2016-normalising,
    title = "Normalising Medical Concepts in Social Media Texts by Learning Semantic Representation",
    author = "Limsopatham, Nut  and
      Collier, Nigel",
    booktitle = "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P16-1096",
    doi = "10.18653/v1/P16-1096",
    pages = "1014--1023",
}
"""

_DESCRIPTION = """
The AskAPatient dataset contains medical concepts written on social media \
mapped to how they are formally written in medical ontologies (SNOMED-CT and AMT).
"""

_HOMEPAGE = "https://zenodo.org/record/55013"

_LICENSE = "CC BY 4.0"

_URLs = "https://zenodo.org/record/55013/files/datasets.zip"

_SUPPORTED_TASKS = [Tasks.PARAPHRASING]  # TODO - (phrase) normalization
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class AskAPatient(datasets.GeneratorBasedBuilder):
    """AskAPatient: Dataset for Normalising Medical Concepts in Social Media Text."""

    DEFAULT_CONFIG_NAME = "ask_a_patient_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="ask_a_patient_source",
            version=SOURCE_VERSION,
            description="AskAPatient source schema",
            schema="source",
            subset_id="ask_a_patient",
        ),
        BigBioConfig(
            name="ask_a_patient_bigbio_t2t",
            version=BIGBIO_VERSION,
            description="AskAPatient simplified BigBio schema",
            schema="bigbio_t2t",
            subset_id="ask_a_patient",
        ),
    ]

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "fold": datasets.Value("int32"),
                    "cui": datasets.Value("string"),
                    "medical_concept": datasets.Value("string"),
                    "social_media_text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "bigbio_t2t":
            features = schemas.text2text_features
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    
    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URLs)
        dataset_dir = os.path.join(dl_dir, "datasets", "AskAPatient")
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "filepaths": glob.glob(
                        os.path.join(dataset_dir, f"AskAPatient.fold-*.{split}.txt")
                    ),
                    "split": split
                },
            )
            for split in [
                datasets.Split.TRAIN,
                datasets.Split.VALIDATION,
                datasets.Split.TEST,
            ]
        ]

    def _generate_examples(self, filepaths, split):
        for filepath in filepaths:
            fold_id = re.search("AskAPatient\.fold-(\d)\.", filepath).group(1)
            with open(filepath, "r", encoding="latin-1") as f:
                document_id = f"{split}_{fold_id}"
                for i, line in enumerate(f):
                    id = f"{document_id}_{i}"
                    cui, medical_concept, social_media_text = line.strip().split("\t")
                    if self.config.schema == "source":
                        yield id, {
                            "fold": int(fold_id),
                            "cui": cui,
                            "medical_concept": medical_concept,
                            "social_media_text": social_media_text,
                        }
                    elif self.config.schema == "bigbio_t2t":
                        # TODO - how to include CUI?
                        yield id, {
                            "id": id, 
                            "document_id": document_id,
                            "text_1": medical_concept,
                            "text_2": social_media_text,
                            "text_1_name": "medical_concept",
                            "text_2_name": "social_media_text",
                        }