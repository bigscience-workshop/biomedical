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
A dataset loader for the SciCite dataset.

SciCite is a dataset of 11K manually annotated citation intents based on
citation context in the computer science and biomedical domains.

Some of the code in this module is based on the corresponding module in the
datasets library.
https://github.com/huggingface/datasets/blob/master/datasets/scicite/scicite.py

In the source schema, we follow the datasets implementation and replace
missing values.
TODO: Use standard BigBio missing values.
"""

import json
import os
from typing import Dict, List, Tuple

import datasets
import numpy as np

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = False
_CITATION = """\
@inproceedings{cohan:naacl19,
  author    = {Arman Cohan and Waleed Ammar and Madeleine van Zuylen and Field Cady},
  title     = {Structural Scaffolds for Citation Intent Classification in Scientific Publications},
  booktitle = {Conference of the North American Chapter of the Association for Computational Linguistics},
  year      = {2019},
  url       = {https://aclanthology.org/N19-1361/},
  doi       = {10.18653/v1/N19-1361},
}
"""

_DATASETNAME = "scicite"

_DESCRIPTION = """\
SciCite is a dataset of 11K manually annotated citation intents based on
citation context in the computer science and biomedical domains.
"""

_HOMEPAGE = "https://allenai.org/data/scicite"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class SciciteDataset(datasets.GeneratorBasedBuilder):
    """SciCite is a dataset of 11K manually annotated citation intents based on
    citation context in the computer science and biomedical domains."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('scicite', name='source')
    # ds_bigbio = datasets.load_dataset('scicite', name='bigbio')

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="scicite_source",
            version=SOURCE_VERSION,
            description="SciCite source schema",
            schema="source",
            subset_id="scicite",
        ),
        BigBioConfig(
            name="scicite_bigbio_text",
            version=BIGBIO_VERSION,
            description="SciCite BigBio schema",
            schema="bigbio_text",
            subset_id="scicite",
        ),
    ]

    DEFAULT_CONFIG_NAME = "scicite_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "source": datasets.Value("string"),
                    "citeStart": datasets.Value("int64"),
                    "sectionName": datasets.Value("string"),
                    "string": datasets.Value("string"),
                    "citeEnd": datasets.Value("int64"),
                    "label": datasets.features.ClassLabel(
                        names=["method", "background", "result"]
                    ),
                    "label_confidence": datasets.Value("float"),
                    "label2": datasets.features.ClassLabel(
                        names=["supportive", "not_supportive", "cant_determine", "none"]
                    ),
                    "label2_confidence": datasets.Value("float"),
                    "citingPaperId": datasets.Value("string"),
                    "citedPaperId": datasets.Value("string"),
                    "isKeyCitation": datasets.Value("bool"),
                    "id": datasets.Value("string"),
                    "unique_id": datasets.Value("string"),
                    "excerpt_index": datasets.Value("int64"),
                }
            )
        elif self.config.schema == "bigbio_text":
            features = schemas.text.features
        else:
            raise ValueError("Unrecognized schema: %s" % self.config.schema)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
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
                    "filepath": os.path.join(data_dir, "scicite", "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "scicite", "test.jsonl"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "scicite", "dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as data_file:
            examples = [json.loads(line) for line in data_file]

            # Preprocesses examples
            keys = set()
            for example in examples:
                # Fixes duplicate keys
                if example["unique_id"] in keys:
                    example["unique_id"] = example["unique_id"] + "_duplicate"
                else:
                    keys.add(example["unique_id"])

            if self.config.schema == "source":
                for example in examples:
                    yield str(example["unique_id"]), {
                        "string": example["string"],
                        "label": str(example["label"]),
                        "sectionName": str(example["sectionName"]),
                        "citingPaperId": str(example["citingPaperId"]),
                        "citedPaperId": str(example["citedPaperId"]),
                        "excerpt_index": int(example["excerpt_index"]),
                        "isKeyCitation": bool(example["isKeyCitation"]),
                        "label2": str(example.get("label2", "none")),
                        "citeEnd": _safe_int(example["citeEnd"]),
                        "citeStart": _safe_int(example["citeStart"]),
                        "source": str(example["source"]),
                        "label_confidence": float(
                            example.get("label_confidence", np.nan)
                        ),
                        "label2_confidence": float(
                            example.get("label2_confidence", np.nan)
                        ),
                        "id": str(example["id"]),
                        "unique_id": str(example["unique_id"]),
                    }

            elif self.config.schema == "bigbio_text":
                for example in examples:
                    if "label2" in example:
                        labels = [example["label"], example["label2"]]
                    else:
                        labels = [example["label"]]

                    yield str(example["unique_id"]), {
                        "id": example["unique_id"],
                        "document_id": example["citingPaperId"],
                        "text": example["string"],
                        "labels": labels,
                    }


def _safe_int(a):
    try:
        # skip NaNs
        return int(a)
    except ValueError:
        return -1
