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
Dataset for medical question summarization introduced in the ACL 2019 paper "On the Summarization of Consumer Health
Questions". Question understanding is one of the main challenges in question answering. In real world applications,
users often submit natural language questions that are longer than needed and include peripheral information that
increases the complexity of the question, leading to substantially more false positives in answer retrieval. In this
paper, we study neural abstractive models for medical question summarization. We introduce the MeQSum corpus of 1,000
summarized consumer health questions.
"""

import os
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from .bigbiohub import text2text_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ['English']
_PUBMED = False
_LOCAL = False
_CITATION = """\
@inproceedings{ben-abacha-demner-fushman-2019-summarization,
    title = "On the Summarization of Consumer Health Questions",
    author = "Ben Abacha, Asma  and
      Demner-Fushman, Dina",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1215",
    doi = "10.18653/v1/P19-1215",
    pages = "2228--2234",
    abstract = "Question understanding is one of the main challenges in question answering. In real world applications, users often submit natural language questions that are longer than needed and include peripheral information that increases the complexity of the question, leading to substantially more false positives in answer retrieval. In this paper, we study neural abstractive models for medical question summarization. We introduce the MeQSum corpus of 1,000 summarized consumer health questions. We explore data augmentation methods and evaluate state-of-the-art neural abstractive models on this new task. In particular, we show that semantic augmentation from question datasets improves the overall performance, and that pointer-generator networks outperform sequence-to-sequence attentional models on this task, with a ROUGE-1 score of 44.16{\%}. We also present a detailed error analysis and discuss directions for improvement that are specific to question summarization.",
}
"""

_DATASETNAME = "meqsum"
_DISPLAYNAME = "MeQSum"

_DESCRIPTION = """\
Dataset for medical question summarization introduced in the ACL 2019 paper "On the Summarization of Consumer Health
Questions". Question understanding is one of the main challenges in question answering. In real world applications,
users often submit natural language questions that are longer than needed and include peripheral information that
increases the complexity of the question, leading to substantially more false positives in answer retrieval. In this
paper, we study neural abstractive models for medical question summarization. We introduce the MeQSum corpus of 1,000
summarized consumer health questions.
"""

_HOMEPAGE = "https://github.com/abachaa/MeQSum"

_LICENSE = 'License information unavailable'

_URLS = {
    _DATASETNAME: "https://github.com/abachaa/MeQSum/raw/master/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx",
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MeQSumDataset(datasets.GeneratorBasedBuilder):
    """Dataset containing 1000 summarized consumer health questions."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="meqsum_source",
            version=SOURCE_VERSION,
            description="MeQSum source schema",
            schema="source",
            subset_id="meqsum",
        ),
        BigBioConfig(
            name="meqsum_bigbio_t2t",
            version=BIGBIO_VERSION,
            description="MeQSum BigBio schema",
            schema="bigbio_t2t",
            subset_id="meqsum",
        ),
    ]

    DEFAULT_CONFIG_NAME = "meqsum_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "File": datasets.Value("string"),
                    "CHQ": datasets.Value("string"),
                    "Summary": datasets.Value("string"),
                }
            )
        elif self.config.schema == "bigbio_t2t":
            features = text2text_features

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
        file_path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(file_path),
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        corpus = pd.read_excel(filepath)

        if self.config.schema == "source":
            for idx, example in corpus.iterrows():
                yield idx, example.to_dict()

        elif self.config.schema == "bigbio_t2t":
            corpus["id"] = corpus.index
            corpus.rename(
                columns={"File": "document_id", "CHQ": "text_1", "Summary": "text_2"},
                inplace=True,
            )
            corpus["text_1_name"] = ""
            corpus["text_2_name"] = ""
            for idx, example in corpus.iterrows():
                yield example["id"], example.to_dict()
