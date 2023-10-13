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

import json
from typing import List, Tuple, Dict
from pathlib import Path

import datasets
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

from .bigbiohub import entailment_features

_LOCAL = False

_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "sem_eval_2024_task_2"

_DESCRIPTION = """\
(Copied from dataset homepage)
## Dataset
The statements and evidence are generated by clinical domain experts, clinical trial organisers, and research oncologists from the Cancer Research UK Manchester Institute and the Digital Experimental Cancer Medicine Team. There are a total of (TBD) statements split evenly across the different sections and classes.
## Description
Each Clinical Trial Report (CTR) consists of 4 sections:
Eligibility criteria - A set of conditions for patients to be allowed to take part in the clinical trial
Intervention - Information concerning the type, dosage, frequency, and duration of treatments being studied.
Results - Number of participants in the trial, outcome measures, units, and the results.
Adverse events - These are signs and symptoms observed in patients during the clinical trial.
For this task, each CTR may contain 1-2 patient groups, called cohorts or arms. These groups may receive different treatments, or have different baseline characteristics.
"""

_HOMEPAGE = "https://sites.google.com/view/nli4ct/semeval-2024?authuser=0"

_LICENSE = ""


_URLS = {
    "train": "https://github.com/ai-systems/Task-2-SemEval-2024/raw/main/training_data.zip",
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


def _get_text(raw_ct_data, section):
    return "".join(raw_ct_data[section])



class SemEval2024Task2Dataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')


    BUILDER_CONFIGS = [
        BigBioConfig(
            name="sem_eval_2024_task_2_source",
            version=SOURCE_VERSION,
            description="sem_eval_2024_task_2 source schema",
            schema="source",
            subset_id="sem_eval_2024_task_2",
        ),
        BigBioConfig(
            name="sem_eval_2024_task_2_ct",
            version=SOURCE_VERSION,
            description="sem_eval_2024_task_2 raw clinical trial data",
            schema="ct",
            subset_id="sem_eval_2024_task_2_ct",
        ),
        BigBioConfig(
            name="sem_eval_2024_task_2_bigbio_TE",
            version=BIGBIO_VERSION,
            description="sem_eval_2024_task_2 BigBio schema",
            schema="bigbio_TE",
            subset_id="sem_eval_2024_task_2",
        ),
    ]

    DEFAULT_CONFIG_NAME = "sem_eval_2024_task_2_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "type": datasets.Value("string"),
                    "section_id": datasets.Value("string"),
                    "primary_id": datasets.Value("string"),
                    "secondary_id": datasets.Value("string"),
                    "statement": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "primary_evidence_index": [datasets.Value("int64")],
                    "secondary_evidence_index": [datasets.Value("int64")],
                }
            )
        elif self.config.schema == "ct":

            features = datasets.Features(
                {
                    "clinical_trial_id": datasets.Value("string"),
                    "intervention": [datasets.Value("string")],
                    "eligibility": [datasets.Value("string")],
                    "results": [datasets.Value("string")],
                    "adverse_events": [datasets.Value("string")],
                }
            )

        elif self.config.schema == "bigbio_TE":
            features = entailment_features

        else:
            raise ValueError(f"Unknown schema {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS["train"]
        data_dir = Path(dl_manager.download_and_extract(urls))

        if self.config.subset_id in {"sem_eval_2024_task_2", "sem_eval_2024_task_2_bigbio_entailment"}: # versions with train/dev split
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # Whatever you put in gen_kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data_dir": data_dir,
                        "split": "train",
                        "config": self.config
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_dir": data_dir,
                        "split": "dev",
                        "config": self.config
                    },
                ),
            ]
        elif self.config.subset_id == "sem_eval_2024_task_2_ct":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_dir": data_dir,
                        "split": "train",
                        "config": self.config
                    },
                )
            ]
        else:
            raise ValueError(f"Unknown subset_id {self.config.subset_id}")


    def _generate_examples(self, data_dir: Path, split: str, config) -> Tuple[int, Dict]:
        """Yields examples as (id, example) tuples."""
        if self.config.schema == "source":
            with open(data_dir / f"training_data/{split}.json", "r") as f:
                raw_data = json.load(f)
            for id_ in sorted(raw_data):
                data_dict = {k.lower().replace(" ", "_"): v for k, v in raw_data[id_].items()} # make keys align with schema

                # add optional keys
                if "secondary_id" not in data_dict:
                    data_dict["secondary_id"] = ""

                if "secondary_evidence_index" not in data_dict:
                    data_dict["secondary_evidence_index"] = []

                yield id_, data_dict

        elif self.config.schema == "ct": # yield only raw clinical trial data
            ct_files = sorted((data_dir / "training_data" / "CT json").glob("*.json"))
            for ct_file in ct_files:
                with open(ct_file, "r") as f:
                    raw_data = json.load(f)
                    id_ = ct_file.stem
                    # make keys align with schema
                    data_dict = {k.lower().replace(" ", "_"): v for k, v in raw_data.items()}
                    yield id_, data_dict

        elif self.config.schema == "bigbio_TE": # combine labels and clinical trial text data here
            with open(data_dir / f"training_data/{split}.json", "r") as f:
                raw_label_data = json.load(f)

            for id_ in sorted(raw_label_data):
                primary_id = raw_label_data[id_]["Primary_id"]
                secondary_id = raw_label_data[id_].get("Secondary_id")

                with open(data_dir / f"training_data/CT json" / f"{primary_id}.json", "r") as f:
                    raw_ct_data = json.load(f)

                text_primary = _get_text(raw_ct_data, section=raw_label_data[id_]["Section_id"])

                if secondary_id:
                    with open(data_dir / f"training_data/CT json" / f"{secondary_id}.json", "r") as f:
                        raw_ct_data = json.load(f)
                    text_secondary = _get_text(raw_ct_data, section=raw_label_data[id_]["Section_id"])
                else:
                    text_secondary = ""

                premise = f"Primary: {text_primary}\n\nSecondary: {text_secondary}"

                yield id_, {"id": id_, "premise": premise, "hypothesis": raw_label_data[id_]["Statement"], "label": raw_label_data[id_]["Label"]}
        else:
            raise ValueError(f"Unknown schema {self.config.schema}")
