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
"""

import os
from typing import List, Tuple, Dict

import datasets
import json
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks, BigBioValues

_CITATION = """\
@inproceedings{,
  author    = {Arman Cohan and Waleed Ammar and Madeleine van Zuylen and Field Cady},
  title     = {Structural Scaffolds for Citation Intent Classification in Scientific Publications},
  booktitle = {Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
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

_LICENSE = ""

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
                    "label": datasets.Value("string"),
                    "label_confidence": datasets.Value("float"),
                    "label2": datasets.Value("string"),
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
            features = schemas.text

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

        with open(filepath, 'r') as data_file:
            examples = [json.loads(line) for line in data_file]

            # Preprocesses examples
            keys = set()
            for example in examples:
                # Sets fields that might not exist to None
                if "label_confidence" not in example:
                    example["label_confidence"] = BigBioValues.NULL
                if "label2" not in example:
                    example["label2"] = BigBioValues.NULL
                if "label2_confidence" not in example:
                    example["label2_confidence"] = BigBioValues.NULL

                # Fixes duplicates
                if example["unique_id"] in keys:
                    example["unique_id"] = example["unique_id"] + "_duplicate"
                else:
                    keys.add(example["unique_id"])

            if self.config.schema == "source":
                for example in examples:
                    yield example["unique_id"], example

            elif self.config.schema == "bigbio_text":
                for example in examples:
                    if example["label2"] != BigBioValues.NULL:
                        labels = [example["label"], example["label2"]]
                    else:
                        labels = [example["label"]]

                    yield example["unique_id"], {
                        "id": example["unique_id"],
                        "document_id": example["citingPaperId"],
                        "text": example["string"],
                        "labels": labels,
                    }


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
