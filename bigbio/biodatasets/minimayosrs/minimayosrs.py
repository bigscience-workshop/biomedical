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
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.CONCEPT]
_LANGUAGES = [Lang.EN]
_PUBMED = False
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

_DATASETNAME = "minimayosrs"

_DESCRIPTION = """\
MiniMayoSRS is a subset of the MayoSRS and consists of 30 term pairs on which a higher inter-annotator agreement was
achieved. The average correlation between physicians is 0.68. The average correlation between medical coders is 0.78.
"""

_HOMEPAGE = "https://conservancy.umn.edu/handle/11299/196265"

_LICENSE = Licenses.CC0_1p0

_URLS = {
    _DATASETNAME: "https://conservancy.umn.edu/bitstream/handle/11299/196265/MiniMayoSRS.csv?sequence=2&isAllowed=y"
}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MinimayosrsDataset(datasets.GeneratorBasedBuilder):
    """MiniMayoSRS is a subset of the MayoSRS and consists of 30 term pairs on which a higher inter-annotator agreement
    was achieved. The average correlation between physicians is 0.68. The average correlation between medical coders
    is 0.78.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="minimayosrs_source",
            version=SOURCE_VERSION,
            description="MiniMayoSRS source schema",
            schema="source",
            subset_id="minimayosrs",
        ),
        BigBioConfig(
            name="minimayosrs_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="MiniMayoSRS BigBio schema",
            schema="bigbio_pairs",
            subset_id="minimayosrs",
        ),
    ]

    DEFAULT_CONFIG_NAME = "minimayosrs_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "code_1": datasets.Value("string"),
                    "code_2": datasets.Value("string"),
                    "label_physicians": datasets.Value("float32"),
                    "label_coders": datasets.Value("float32"),
                }
            )

        elif self.config.schema == "bigbio_pairs":
            features = schemas.pairs_features

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
        filepath = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": filepath},
            )
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        data = pd.read_csv(
            filepath,
            sep=",",
            header=0,
            names=[
                "label_physicians",
                "label_coders",
                "code_1",
                "code_2",
                "text_1",
                "text_2",
            ],
        )

        if self.config.schema == "source":
            for id_, row in data.iterrows():
                yield id_, row.to_dict()

        elif self.config.schema == "bigbio_pairs":
            for id_, row in data.iterrows():
                yield id_, {
                    "id": id_,
                    "document_id": id_,
                    "text_1": row["text_1"],
                    "text_2": row["text_2"],
                    "label": str((row["label_physicians"] + row["label_coders"]) / 2),
                }
