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

import os
from typing import List

import datasets

from bigbio.utils import schemas

from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

import json

_LOCAL = False
_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2010.14235,
  doi = {10.48550/ARXIV.2010.14235},
  
  url = {https://arxiv.org/abs/2010.14235},
  
  author = {Lu, Yao and Dong, Yue and Charlin, Laurent},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Multi-XScience: A Large-scale Dataset for Extreme Multi-document Summarization of Scientific Articles},
  
  publisher = {arXiv},
  
  year = {2020},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""

_DATASETNAME = "multi_xscience"

_DESCRIPTION = """\
Multi-document summarization is a challenging task for which there exists little large-scale datasets. 
We propose Multi-XScience, a large-scale multi-document summarization dataset created from scientific articles. 
Multi-XScience introduces a challenging multi-document summarization task: writing the related-work section 
of a paper based on its abstract and the articles it references. Our work is inspired by extreme summarization, 
a dataset construction protocol that favours abstractive modeling approaches. Descriptive statistics and 
empirical results---using several state-of-the-art models trained on the Multi-XScience dataset---reveal t
hat Multi-XScience is well suited for abstractive models.
"""

_HOMEPAGE = "https://github.com/yaolu/Multi-XScience"

_LICENSE = "MIT License"

_URLS = {
    _DATASETNAME: [
        "https://github.com/yaolu/Multi-XScience/blob/master/data/train.json.gz?raw=true",
        "https://github.com/yaolu/Multi-XScience/blob/master/data/test.json.gz?raw=true",
        "https://github.com/yaolu/Multi-XScience/blob/master/data/val.json.gz?raw=true",
    ],
}

_SUPPORTED_TASKS = [Tasks.PARAPHRASING, Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MultiXScience(datasets.GeneratorBasedBuilder):
    """
    Dataset for the EMNLP 2020 paper, Multi-XScience:
    A Large-scale Dataset for Extreme Multi-document Summarization
    of Scientific Articles.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="multi_xscience_source",
            version=SOURCE_VERSION,
            description="multi_xscience source schema",
            schema="source",
            subset_id="multi_xscience",
        ),
        BigBioConfig(
            name="multi_xscience_bigbio_t2t",
            version=BIGBIO_VERSION,
            description="multi_xscienceBigBio schema",
            schema="bigbio_t2t",
            subset_id="multi_xscience",
        ),
    ]

    DEFAULT_CONFIG_NAME = "multi_xscience_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "aid": datasets.Value("string"),
                    "mid": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "ref_abstract": datasets.Sequence(
                        {
                            "mid": datasets.Value("string"),
                            "abstract": datasets.Value("string"),
                        }
                    ),
                }
            )
        elif self.config.schema == "bigbio_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir[0]).replace("\\", "/"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir[1]).replace("\\", "/"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir[2]).replace("\\", "/"),
                    "split": "val",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    def _generate_examples(self, filepath, split):
        j_file = open(filepath, "r")
        j_file.seek(0)
        j_json = json.load(j_file)

        if self.config.schema == "source":
            for key, example in enumerate(j_json):
                yield key, {
                    "aid": example["aid"],
                    "mid": example["mid"],
                    "abstract": example["abstract"],
                    "ref_abstract": [
                        {
                            "mid": example["ref_abstract"][key]["mid"],
                            "abstract": example["ref_abstract"][key]["abstract"],
                        }
                        for key in example["ref_abstract"].keys()
                    ],
                }

        elif self.config.schema == "bigbio_t2t":
            uid = 0

            for key, example in enumerate(j_json):
                uid += 1
                yield key, {
                    "id": str(uid),
                    "document_id": str(key),
                    "text_1": example["abstract"],
                    "text_2": " ".join(
                        [e["abstract"]
                            for e in example["ref_abstract"].values()]
                    ),
                    "text_1_name": "Abstract of query paper",
                    "text_2_name": "Cite abstracts",
                }

        j_file.close()
