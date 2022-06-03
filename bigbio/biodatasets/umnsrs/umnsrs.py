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
UMNSRS, developed by Pakhomov, et al., consists of 725 clinical term pairs whose semantic similarity and relatedness.
The similarity and relatedness of each term pair was annotated based on a continuous scale by having the resident touch
a bar on a touch sensitive computer screen to indicate the degree of similarity or relatedness. The Intraclass
Correlation Coefficient (ICC) for the reference standard tagged for similarity was 0.47, and 0.50 for relatedness.
Therefore, as suggested by Pakhomov and colleagues, the subset below consists of 401 pairs for the similarity set and
430 pairs for the relatedness set which each have an ICC equal to 0.73.
"""

from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = False
_CITATION = """\
@inproceedings{pakhomov2010semantic,
  title={Semantic similarity and relatedness between clinical terms: an experimental study},
  author={Pakhomov, Serguei and McInnes, Bridget and Adam, Terrence and Liu, Ying and Pedersen, Ted and Melton, \
  Genevieve B},
  booktitle={AMIA annual symposium proceedings},
  volume={2010},
  pages={572},
  year={2010},
  organization={American Medical Informatics Association}
}
"""

_DATASETNAME = "umnsrs"

_DESCRIPTION = """\
UMNSRS, developed by Pakhomov, et al., consists of 725 clinical term pairs whose semantic similarity and relatedness.
The similarity and relatedness of each term pair was annotated based on a continuous scale by having the resident touch
a bar on a touch sensitive computer screen to indicate the degree of similarity or relatedness.
The following subsets are available:
- similarity: A set of 566 UMLS concept pairs manually rated for semantic similarity (e.g. whale-dolphin) using a
  continuous response scale.
- relatedness: A set of 588 UMLS concept pairs manually rated for semantic relatedness (e.g. needle-thread) using a
  continuous response scale.
- similarity_mod: Modification of the UMNSRS-Similarity dataset to exclude control samples and those pairs that did not
  match text in clinical, biomedical and general English corpora. Exact modifications are detailed in the paper (Corpus
  Domain Effects on Distributional Semantic Modeling of Medical Terms. Serguei V.S. Pakhomov, Greg Finley, Reed McEwan,
  Yan Wang, and Genevieve B. Melton. Bioinformatics. 2016; 32(23):3635-3644). The resulting dataset contains 449 pairs.
- relatedness_mod: Modification of the UMNSRS-Relatedness dataset to exclude control samples and those pairs that did
  not match text in clinical, biomedical and general English corpora. Exact modifications are detailed in the paper
  (Corpus Domain Effects on Distributional Semantic Modeling of Medical Terms. Serguei V.S. Pakhomov, Greg Finley,
  Reed McEwan, Yan Wang, and Genevieve B. Melton. Bioinformatics. 2016; 32(23):3635-3644).
  The resulting dataset contains 458 pairs.
"""

_HOMEPAGE = "https://conservancy.umn.edu/handle/11299/196265/"

_LICENSE = Licenses.CC0_1p0

_BASE_URL = "https://conservancy.umn.edu/bitstream/handle/11299/196265/"

_URLS = {
    "umnsrs_similarity": _BASE_URL + "UMNSRS_similarity.csv",
    "umnsrs_relatedness": _BASE_URL + "UMNSRS_relatedness.csv",
    "umnsrs_similarity_mod": _BASE_URL + "UMNSRS_similarity_mod449_word2vec.csv",
    "umnsrs_relatedness_mod": _BASE_URL + "UMNSRS_relatedness_mod458_word2vec.csv",
}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class UmnsrsDataset(datasets.GeneratorBasedBuilder):
    """UMNSRS, developed by Pakhomov, et al., contains clinical term pairs whose semantic similarity and
    relatedness were scored by experts."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    for subset in ["similarity", "relatedness"]:
        for mod in ["_mod", ""]:
            BUILDER_CONFIGS.append(
                BigBioConfig(
                    name=f"umnsrs_{subset}{mod}_source",
                    version=SOURCE_VERSION,
                    description=f"UMNSRS {subset}{mod} source schema",
                    schema="source",
                    subset_id=f"umnsrs_{subset}{mod}",
                )
            )
            BUILDER_CONFIGS.append(
                BigBioConfig(
                    name=f"umnsrs_{subset}{mod}_bigbio_pairs",
                    version=BIGBIO_VERSION,
                    description=f"UMNSRS {subset}{mod} BigBio schema",
                    schema="bigbio_pairs",
                    subset_id=f"umnsrs_{subset}{mod}",
                )
            )

    DEFAULT_CONFIG_NAME = "umnsrs_similarity_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "mean_score": datasets.Value("float32"),
                    "std_score": datasets.Value("float32"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
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

        urls = _URLS[self.config.subset_id]
        filepath = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                },
            )
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        print(filepath)
        data = pd.read_csv(
            filepath,
            sep=",",
            header=0,
            names=["mean_score", "std_score", "text_1", "text_2", "code_1", "code_2"],
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
                    "label": row["mean_score"],
                }
