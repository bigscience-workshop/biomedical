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

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.license import Licenses
from bigbio.utils.constants import Tasks

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

_LICENSE_OLD = "MIT"

_URLS = {
    _DATASETNAME: "http://evidence-inference.ebm-nlp.com/v2.0.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "2.0.0"

_BIGBIO_VERSION = "1.0.0"


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
            name="evidence-inference_bigbio_te",
            version=BIGBIO_VERSION,
            description="evidence-inference BigBio schema",
            schema="bigbio_te",
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

        elif self.config.schema == "bigbio_te":
            features = schemas.entailment_features

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

    def _generate_examples(self, filepaths, datapath, split, datadir) -> Tuple[int, Dict]:
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

        if self.config.schema == "source":
            for key, sample in prompts.iterrows():
                pid = sample["PromptID"]
                pmcid = sample["PMCID"]
                label = lookup(annotations, pid, "Label")
                start = lookup(evidences, pmcid, "Evidence Start")
                end = lookup(evidences, pmcid, "Evidence End")

                if label == -1:
                    continue

                evidence = extract_evidence(pmcid, start, end)

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

        elif self.config.schema == "bigbio_te":
            for key, sample in prompts.iterrows():
                pid = sample["PromptID"]
                pmcid = sample["PMCID"]
                label = lookup(annotations, pid, "Label")
                start = lookup(evidences, pmcid, "Evidence Start")
                end = lookup(evidences, pmcid, "Evidence End")

                if label == -1:
                    continue

                evidence = extract_evidence(pmcid, start, end)

                feature_dict = {
                    "id": uid,
                    "premise": "\t".join([sample["Intervention"], sample["Comparator"], sample["Outcome"]]),
                    "hypothesis": evidence,
                    "label": label,
                }

                uid += 1
                yield key, feature_dict
