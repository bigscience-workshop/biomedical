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
MayoSRS consists of 101 clinical term pairs whose relatedness was determined by
nine medical coders and three physicians from the Mayo Clinic.
"""

from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{pedersen2007measures,
  title={Measures of semantic similarity and relatedness in the biomedical domain},
  author={Pedersen, Ted and Pakhomov, Serguei VS and Patwardhan, Siddharth and Chute, Christopher G},
  journal={Journal of biomedical informatics},
  volume={40},
  number={3},
  pages={288--299},
  year={2007},
  publisher={Elsevier}
}
"""

_DATASETNAME = "mayosrs"

_DESCRIPTION = """\
MayoSRS consists of 101 clinical term pairs whose relatedness was determined by
nine medical coders and three physicians from the Mayo Clinic.
"""

_HOMEPAGE = "https://conservancy.umn.edu/handle/11299/196265"

_LICENSE = "CC0 1.0 Universal"

_URLS = {_DATASETNAME: "https://conservancy.umn.edu/bitstream/handle/11299/196265/MayoSRS.csv?sequence=1&isAllowed=y"}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MayosrsDataset(datasets.GeneratorBasedBuilder):
    """MayoSRS consists of 101 clinical term pairs whose relatedness was
    determined by nine medical coders and three physicians from the Mayo Clinic."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="mayosrs_source",
            version=SOURCE_VERSION,
            description="MayoSRS source schema",
            schema="source",
            subset_id="mayosrs",
        ),
        BigBioConfig(
            name="mayosrs_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="MayoSRS BigBio schema",
            schema="bigbio_pairs",
            subset_id="mayosrs",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mayosrs_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "label": datasets.Value("float32"),
                    "code_1": datasets.Value("string"),
                    "code_2": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_pairs":
            features = schemas.pairs_features

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
        filepath = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if split == "train":

            data = pd.read_csv(filepath, sep=",", header=0, names=["label", "code_1", "code_2", "text_1", "text_2"])

            if self.config.schema == "source":
                for id_, row in data.iterrows():
                    yield id_, row.to_dict()

            elif self.config.schema == "bigbio_pairs":
                for id_, row in data.iterrows():
                    yield id_, {
                        "id": id_,  # uid is an unique identifier for every record that starts from 1
                        "document_id": id_,
                        "text_1": row["text_1"],
                        "text_2": row["text_2"],
                        "label": str(row["label"]),
                    }

        else:
            print("There's no test/val split available for the given dataset")
            return
