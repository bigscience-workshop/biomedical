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
A dataset loading script for the CODIESP corpus.

The CODIESP dataset is a collection of 1,000 manually selected clinical 
case studies in Spanish that was designed for the Clinical Case Coding 
in Spanish Shared Task, as part of the CLEF 2020 conference. This community
task was divided into 3 sub-tasks: diagnosis coding (CodiEsp-D), procedure
coding (CodiEsp-P) and Explainable AI (CodiEsp-X).
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@inproceedings{
  title={Overview of automatic clinical coding: annotations, guidelines, and \
  solutions for non-english clinical cases at codiesp track of CLEF eHealth 2020},
  author={Miranda-Escalada, Antonio and Gonzalez-Agirre, Aitor and Armengol-Estap{\'e}, Jordi and Krallinger, Martin},
  booktitle={Working Notes of Conference and Labs of the Evaluation (CLEF) Forum. CEUR Workshop Proceedings},
  year={2020}
}
"""

_DATASETNAME = "cantemist"

_DESCRIPTION = """\
Synthetic corpus of 1,000 manually selected clinical case studies in Spanish that was designed for \
the Clinical Case Coding in Spanish Shared Task, as part of the CLEF 2020 conference.

The goal of the task was to automatically assign ICD10 codes (CIE-10, in Spanish) to clinical case documents, 
being evaluated against manually generated ICD10 codifications. The CodiEsp corpus was selected manually by \
practicing physicians and clinical documentalists and annotated by clinical coding professionals meeting strict \
quality criteria. They reached an inter-annotator agreement of 88.6\% for diagnosis coding, 88.9\% for procedure \
coding and 80.5\% for the textual reference annotation.

The final collection of 1,000 clinical cases that make up the corpus had a total of 16,504 sentences and \
396,988 words. All documents are in Spanish language and CIE10 is the coding terminology (the Spanish version \
of ICD10-CM and ICD10-PCS). The CodiEsp corpus has been randomly sampled into three subsets. \
The train set contains 500 clinical cases, while the development and test sets have 250 clinical cases each. 

The CodiEsp track was divided into three sub-tracks (2 main and 1 exploratory):

CodiEsp-D: The Diagnosis Coding sub-task, which requires automatic ICD10-CM [CIE10-Diagnóstico] code assignment. 

CodiEsp-P: The Procedure Coding sub-task, which requires automatic ICD10-PCS [CIE10-Procedimiento] code assignment.

CodiEsp-X: The Explainable AI exploratory sub-task, which requires to submit the reference to the predicted codes \
(both ICD10-CM and ICD10-PCS). The goal of this novel task was not only to predict the correct codes but also to \
present the reference in the text that supports the code predictions.

For further information, please visit https://temu.bsc.es/codiesp or send an email to encargo-pln-life@bsc.es
"""

_HOMEPAGE = "https://temu.bsc.es/codiesp/index.php/2019/09/19/datasets/"

_LICENSE = "Creative Commons Attribution 4.0 International"

_URLS = {
    "codiesp": "https://zenodo.org/record/3837305/files/codiesp.zip?download=1",
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.4.0"

_BIGBIO_VERSION = "1.0.0"


class CodiespDataset(datasets.GeneratorBasedBuilder):
    """Collection of 1,000 manually selected clinical case studies in Spanish."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="codiesp_source",
            version=SOURCE_VERSION,
            description="CodiEsp source schema",
            schema="source",
            subset_id="codiesp",
        ),
        BigBioConfig(
            name="codiespD_bigbio_text",
            version=BIGBIO_VERSION,
            description="CodiEsp BigBio schema for the Diagnosis Coding subtask",
            schema="bigbio_text",
            subset_id="codiesp_d",
        ),
        BigBioConfig(
            name="codiespP_bigbio_text",
            version=BIGBIO_VERSION,
            description="CodiEsp BigBio schema for the Procedure Coding sub-task",
            schema="bigbio_text",
            subset_id="codiesp_p",
        ),
        BigBioConfig(
            name="codiespX_bigbio_kb",
            version=BIGBIO_VERSION,
            description="CodiEsp BigBio schema for the Explainable AI sub-task",
            schema="bigbio_kb",
            subset_id="codiesp_x",
        ),
    ]

    DEFAULT_CONFIG_NAME = "codiesp_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "task_d_labels": [datasets.Value("string")],
                    "task_p_labels": [datasets.Value("string")],
                    "task_x": [
                        {
                            "label": datasets.Value("string"),
                            "code": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "spans": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                },
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """
        Downloads/extracts the data to generate the train, validation and test splits.

        Each split is created by instantiating a `datasets.SplitGenerator`, which will
        call `this._generate_examples` with the keyword arguments in `gen_kwargs`.
        """

        data_dir = dl_manager.download_and_extract(_URLS["codiesp"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "final_dataset_v4_to_publish/train")),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "final_dataset_v4_to_publish/test")),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": Path(os.path.join(data_dir, "final_dataset_v4_to_publish/dev")),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """
        This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        Method parameters are unpacked from `gen_kwargs` as given in `_split_generators`.
        """

        paths = {"text_files": Path(os.path.join(filepath, "text_files"))}
        for task in ["codiesp_d", "codiesp_p", "codiesp_x"]:
            paths[task] = Path(os.path.join(filepath, f"{split}{task[-1].upper()}.tsv"))

        if self.config.subset_id == "codiesp_d" or self.config.subset_id == "codiesp_p":
            df = pd.read_csv(paths[self.config.subset_id], sep="\t", header=None)

            file_codes_dict = defaultdict(list)
            for idx, row in df.iterrows():
                file, code = row[0], row[1]
                file_codes_dict[file].append(code)

            for guid, (file, codes) in enumerate(file_codes_dict.items()):
                text_file = Path(os.path.join(paths["text_files"], f"{file}.txt"))
                example = {
                    "id": str(guid),
                    "document_id": file,
                    "text": text_file.read_text(),
                    "labels": codes,
                }
                yield guid, example

        elif self.config.subset_id == "codiesp_x":
            df = pd.read_csv(paths[self.config.subset_id], sep="\t", header=None)

            example = {}
            for guid, row in df.iterrows():
                file, label, code, text, spans = row[0], row[1], row[2], row[3], row[4]
                appearances = spans.split(";")
                spans = []
                for a in appearances:
                    spans.append([int(a.split()[0]), int(a.split()[1])])
                example["id"] = str(guid)
                example["document_id"] = file
                example["passages"] = [{"id": label, "type": code, "text": [text], "offsets": spans}]
                example["entities"] = []
                example["events"] = []
                example["coreferences"] = []
                example["relations"] = []
                yield guid, example

        elif self.config.subset_id == "codiesp":
            dfs = {}
            task_dicts = {}
            for task in ["codiesp_d", "codiesp_p", "codiesp_x"]:
                dfs[task] = pd.read_csv(paths[task], sep="\t", header=None)
                file_codes_dict = defaultdict(list)
                if task != "codiesp_x":
                    for idx, row in dfs[task].iterrows():
                        file, code = row[0], row[1]
                        file_codes_dict[file].append(code)
                else:
                    for idx, row in dfs[task].iterrows():
                        file, label, code, text, spans = row[0], row[1], row[2], row[3], row[4]
                        appearances = spans.split(";")
                        spans = []
                        for a in appearances:
                            spans.append([int(a.split()[0]), int(a.split()[1])])
                        file_codes_dict[file].append({"label": label, "code": code, "text": text, "spans": spans})
                task_dicts[task] = file_codes_dict

            # Iterate over the files from the task D dataset, since it is a superset of the other two
            for guid, (file, codes) in enumerate(task_dicts["codiesp_d"].items()):
                example = {
                    "id": guid,
                    "document_id": file,
                    "text": Path(os.path.join(paths["text_files"], f"{file}.txt")).read_text(),
                    "task_d_labels": codes,
                    "task_p_labels": [],
                    "task_x": {"label": "", "code": "", "text": "", "spans": []},
                }

                if file in task_dicts["codiesp_p"]:
                    example["task_p_labels"] = task_dicts["codiesp_p"][file]

                if file in task_dicts["codiesp_x"]:
                    example["task_x"] = task_dicts["codiesp_x"][file]

                yield guid, example

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
