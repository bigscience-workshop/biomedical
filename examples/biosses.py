# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
BioSSES computes similarity of biomedical sentences by utilizing WordNet as the
general domain ontology and UMLS as the biomedical domain specific ontology.
The original paper outlines the approaches with respect to using annotator
score as golden standard. Source view will return all annotator score
individually whereas the Bigbio view will return the mean of the annotator
score.

Note: The original files are Word documents, compressed using RAR. This data
loader uses a version that privides the same data in text format.
"""
import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks, Tags
from bigbio.utils.license import Licenses

_DATASETNAME = "biosses"

_TAGS = []
_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = False
_CITATION = """
@article{souganciouglu2017biosses,
  title={BIOSSES: a semantic sentence similarity estimation system for the biomedical domain},
  author={Soğancıoğlu, Gizem, Hakime Öztürk, and Arzucan Özgür},
  journal={Bioinformatics},
  volume={33},
  number={14},
  pages={i49--i58},
  year={2017},
  publisher={Oxford University Press}
}
"""

_DESCRIPTION = """
BioSSES computes similarity of biomedical sentences by utilizing WordNet as the
general domain ontology and UMLS as the biomedical domain specific ontology.
The original paper outlines the approaches with respect to using annotator
score as golden standard. Source view will return all annotator score
individually whereas the Bigbio view will return the mean of the annotator
score.
"""

_HOMEPAGE = "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html"

_LICENSE = Licenses.GPL_3p0

_URLs = {
    "source": "https://huggingface.co/datasets/bigscience-biomedical/biosses/raw/main/annotation_pairs_scores.tsv",
    "bigbio_pairs": "https://huggingface.co/datasets/bigscience-biomedical/biosses/raw/main/annotation_pairs_scores.tsv",
}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


# The BIOSSES dataset does not provide canonical train/dev/test splits.
# However the BLUE and BLURB datasets use the following split definitions.
# see https://github.com/bigscience-workshop/biomedical/issues/664

TRAIN_INDEXES = [
    78,
    45,
    35,
    50,
    27,
    13,
    87,
    1,
    58,
    99,
    55,
    74,
    66,
    39,
    44,
    18,
    84,
    76,
    19,
    10,
    75,
    46,
    15,
    86,
    60,
    14,
    51,
    79,
    29,
    34,
    94,
    28,
    62,
    42,
    21,
    30,
    11,
    53,
    6,
    12,
    26,
    48,
    31,
    32,
    77,
    37,
    95,
    85,
    36,
    56,
    43,
    61,
    16,
    5,
    67,
    65,
    54,
    3,
    73,
    98,
    17,
    4,
    92,
    93,
]
DEV_INDEXES = [
    88,
    82,
    8,
    63,
    47,
    68,
    40,
    90,
    100,
    24,
    41,
    91,
    80,
    9,
    72,
    2,
]
TEST_INDEXES = [
    59,
    96,
    70,
    22,
    81,
    38,
    57,
    23,
    33,
    89,
    69,
    49,
    7,
    71,
    97,
    25,
    83,
    64,
    52,
    20,
]


class BiossesDataset(datasets.GeneratorBasedBuilder):
    """BIOSSES : Biomedical Semantic Similarity Estimation System"""

    DEFAULT_CONFIG_NAME = "biosses_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="biosses_source",
            version=SOURCE_VERSION,
            description="BIOSSES source schema",
            schema="source",
            subset_id="biosses",
        ),
        BigBioConfig(
            name="biosses_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="BIOSSES simplified BigBio schema",
            schema="bigbio_pairs",
            subset_id="biosses",
        ),
    ]

    def _info(self):

        if self.config.name == "biosses_source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "document_id": datasets.Value("int64"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "annotator_a": datasets.Value("int64"),
                    "annotator_b": datasets.Value("int64"),
                    "annotator_c": datasets.Value("int64"),
                    "annotator_d": datasets.Value("int64"),
                    "annotator_e": datasets.Value("int64"),
                }
            )
        elif self.config.name == "biosses_bigbio_pairs":
            features = schemas.pairs_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        my_urls = _URLs[self.config.schema]
        dl_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir,
                    "split": "train",
                    "indexes": TRAIN_INDEXES,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dl_dir,
                    "split": "validation",
                    "indexes": DEV_INDEXES,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": dl_dir,
                    "split": "test",
                    "indexes": TEST_INDEXES,
                },
            ),
        ]

    def _generate_examples(self, filepath, split, indexes):

        df = pd.read_csv(filepath, sep="\t", encoding="utf-8")
        df = df[df["sentence_id"].isin(indexes)]

        if self.config.schema == "source":
            for uid, row in df.iterrows():
                yield uid, {
                    "id": uid,
                    "document_id": row["sentence_id"],
                    "text_1": row["sentence_1"],
                    "text_2": row["sentence_2"],
                    "annotator_a": row["annotator_a"],
                    "annotator_b": row["annotator_b"],
                    "annotator_c": row["annotator_c"],
                    "annotator_d": row["annotator_d"],
                    "annotator_e": row["annotator_e"],
                }

        elif self.config.schema == "bigbio_pairs":
            for uid, row in df.iterrows():
                yield uid, {
                    "id": uid,
                    "document_id": row["sentence_id"],
                    "text_1": row["sentence_1"],
                    "text_2": row["sentence_2"],
                    "label": str(
                        (
                            row["annotator_a"]
                            + row["annotator_b"]
                            + row["annotator_c"]
                            + row["annotator_d"]
                            + row["annotator_e"]
                        )
                        / 5
                    ),
                }
