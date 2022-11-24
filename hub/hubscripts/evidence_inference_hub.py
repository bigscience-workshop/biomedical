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
The dataset consists of biomedical articles describing randomized control trials (RCTs)
that compare multiple treatments. Each of these articles will have multiple questions,
or 'prompts' associated with them. These prompts will ask about the relationship between
an intervention and comparator with respect to an outcome, as reported in the trial.
For example, a prompt may ask about the reported effects of aspirin as compared to placebo
on the duration of headaches.
For the sake of this task, we assume that a particular article will report that the intervention of interest either
significantly increased, significantly decreased or had significant effect on the outcome, relative to the comparator.
"""

import os
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from .bigbiohub import qa_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ['English']
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{deyoung-etal-2020-evidence,
    title = "Evidence Inference 2.0: More Data, Better Models",
    author = "DeYoung, Jay  and
      Lehman, Eric  and
      Nye, Benjamin  and
      Marshall, Iain  and
      Wallace, Byron C.",
    booktitle = "Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bionlp-1.13",
    pages = "123--132",
}
"""

_DATASETNAME = "evidence_inference"
_DISPLAYNAME = "Evidence Inference 2.0"

_DESCRIPTION = """\
The dataset consists of biomedical articles describing randomized control trials (RCTs) that compare multiple
treatments. Each of these articles will have multiple questions, or 'prompts' associated with them.
These prompts will ask about the relationship between an intervention and comparator with respect to an outcome,
as reported in the trial. For example, a prompt may ask about the reported effects of aspirin as compared
to placebo on the duration of headaches. For the sake of this task, we assume that a particular article
will report that the intervention of interest either significantly increased, significantly decreased
or had significant effect on the outcome, relative to the comparator.
"""

_HOMEPAGE = "https://github.com/jayded/evidence-inference"

_LICENSE = 'MIT License'

_URLS = {
    _DATASETNAME: "http://evidence-inference.ebm-nlp.com/v2.0.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "2.0.0"

_BIGBIO_VERSION = "1.0.0"

QA_CHOICES = [
    "significantly increased",
    "no significant difference",
    "significantly decreased",
]

# Some examples are removed due to comments on the dataset's github page
# https://github.com/jayded/evidence-inference/blob/master/annotations/README.md#caveat

INCORRECT_PROMPT_IDS = set([
    911, 912, 1262, 1261, 3044, 3248, 3111, 3620, 4308, 4490, 4491, 4324,
    4325, 4492, 4824, 5000, 5001, 5002, 5046, 5047, 4948, 5639, 5710, 5752,
    5775, 5782, 5841, 5843, 5861, 5862, 5863, 5964, 5965, 5966, 5975, 4807,
    5776, 5777, 5778, 5779, 5780, 5781, 6034, 6065, 6066, 6666, 6667, 6668,
    6669, 7040, 7042, 7944, 8590, 8605, 8606, 8639, 8640, 8745, 8747, 8749,
    8877, 8878, 8593, 8631, 8635, 8884, 8886, 8773, 10032, 10035, 8876, 8875,
    8885, 8917, 8921, 8118, 10885, 10886, 10887, 10888, 10889, 10890
])

QUESTIONABLE_PROMPT_IDS = set([
    7811, 7812, 7813, 7814, 7815, 8197, 8198, 8199,
    8200, 8201, 9429, 9430, 9431, 8536, 9432
])

SOMEWHAT_MALFORMED_PROMPT_IDS = set([
    3514, 346, 5037, 4715, 8767, 9295, 9297, 8870, 9862
])

SKIP_PROMPT_IDS = INCORRECT_PROMPT_IDS | QUESTIONABLE_PROMPT_IDS | SOMEWHAT_MALFORMED_PROMPT_IDS


class EvidenceInferenceDataset(datasets.GeneratorBasedBuilder):
    f"""{_DESCRIPTION}"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="evidence-inference_source",
            version=SOURCE_VERSION,
            description="evidence-inference source schema",
            schema="source",
            subset_id="evidence-inference",
        ),
        BigBioConfig(
            name="evidence-inference_bigbio_qa",
            version=BIGBIO_VERSION,
            description="evidence-inference BigBio schema",
            schema="bigbio_qa",
            subset_id="evidence-inference",
        ),
    ]

    DEFAULT_CONFIG_NAME = "evidence-inference_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "prompt_id": datasets.Value("int64"),
                    "pmcid": datasets.Value("int64"),
                    "label": datasets.Value("string"),
                    "evidence": datasets.Value("string"),
                    "intervention": datasets.Value("string"),
                    "comparator": datasets.Value("string"),
                    "outcome": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_qa":
            features = qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": [
                        os.path.join(data_dir, "annotations_merged.csv"),
                        os.path.join(data_dir, "prompts_merged.csv"),
                    ],
                    "datapath": os.path.join(data_dir, "txt_files"),
                    "split": "train",
                    "datadir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": [
                        os.path.join(data_dir, "annotations_merged.csv"),
                        os.path.join(data_dir, "prompts_merged.csv"),
                    ],
                    "datapath": os.path.join(data_dir, "txt_files"),
                    "split": "validation",
                    "datadir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": [
                        os.path.join(data_dir, "annotations_merged.csv"),
                        os.path.join(data_dir, "prompts_merged.csv"),
                    ],
                    "datapath": os.path.join(data_dir, "txt_files"),
                    "split": "test",
                    "datadir": data_dir,
                },
            ),
        ]

    def _generate_examples(
        self, filepaths, datapath, split, datadir
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(f"{datadir}/splits/{split}_article_ids.txt", "r") as f:
            ids = [int(i.strip()) for i in f.readlines()]
        prompts = pd.read_csv(filepaths[-1], encoding="utf8")
        prompts = prompts[prompts["PMCID"].isin(ids)]

        annotations = pd.read_csv(filepaths[0], encoding="utf8").set_index("PromptID")
        evidences = pd.read_csv(filepaths[0], encoding="utf8").set_index("PMCID")
        evidences = evidences[evidences["Evidence Start"] != -1]
        uid = 0

        def lookup(df: pd.DataFrame, id, col) -> str:
            try:
                label = df.loc[id][col]
                if isinstance(label, pd.Series):
                    return label.values[0]
                else:
                    return label
            except KeyError:
                return -1

        def extract_evidence(doc_id, start, end):
            p = f"{datapath}/PMC{doc_id}.txt"
            with open(p, "r") as f:
                return f.read()[start:end]


        for key, sample in prompts.iterrows():

            pid = sample["PromptID"]
            pmcid = sample["PMCID"]
            label = lookup(annotations, pid, "Label")
            start = lookup(evidences, pmcid, "Evidence Start")
            end = lookup(evidences, pmcid, "Evidence End")

            if pid in SKIP_PROMPT_IDS:
                continue

            if label == -1:
                continue

            evidence = extract_evidence(pmcid, start, end)

            if self.config.schema == "source":

                feature_dict = {
                    "id": uid,
                    "pmcid": pmcid,
                    "prompt_id": pid,
                    "intervention": sample["Intervention"],
                    "comparator": sample["Comparator"],
                    "outcome": sample["Outcome"],
                    "evidence": evidence,
                    "label": label,
                }

                uid += 1
                yield key, feature_dict

            elif self.config.schema == "bigbio_qa":

                context = evidence
                question = (
                    f"Compared to {sample['Comparator']} "
                    f"what was the result of {sample['Intervention']} on {sample['Outcome']}?"
                )
                feature_dict = {
                    "id": uid,
                    "question_id": pid,
                    "document_id": pmcid,
                    "question": question,
                    "type": "multiple_choice",
                    "choices": QA_CHOICES,
                    "context": context,
                    "answer": [label],
                }

                uid += 1
                yield key, feature_dict
