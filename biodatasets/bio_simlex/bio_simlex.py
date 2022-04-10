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
Bio-SimLex enables intrinsic evaluation of word representations. This evaluation can serve as a predictor of performance
on various downstream tasks in the biomedical domain. The results on Bio-SimLex using standard word representation
models highlight the importance of developing dedicated evaluation resources for NLP in biomedicine for particular
word classes (e.g. verbs).
[bigbio_schema_name] = pairs
"""

import os
from typing import Dict, List, Tuple

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{article,
author = {Chiu, Billy and Pyysalo, Sampo and Vulić, Ivan and Korhonen, Anna},
year = {2018},
month = {02},
pages = {},
title = {Bio-SimVerb and Bio-SimLex: Wide-coverage evaluation sets of word similarity in biomedicine},
volume = {19},
journal = {BMC Bioinformatics},
doi = {10.1186/s12859-018-2039-z}
}{}
}
"""

_DATASETNAME = "bio_simlex"

_DESCRIPTION = """\
Bio-SimLex enables intrinsic evaluation of word representations. This evaluation can serve as a predictor of performance
on various downstream tasks in the biomedical domain. The results on Bio-SimLex using standard word representation
models highlight the importance of developing dedicated evaluation resources for NLP in biomedicine for particular
word classes (e.g. verbs).
"""

_HOMEPAGE = "https://github.com/cambridgeltl/bio-simverb"


_LICENSE = """Open Access This article is distributed under the terms of the
Creative Commons Attribution 4.0 International License
(http://creativecommons.org/licenses/by/4.0/), which permits
unrestricted use, distribution, and reproduction in any medium,
provided you give appropriate credit to the original author(s) and
the source, provide a link to the Creative Commons license, and
indicate if changes were made. The Creative Commons Public Domain
Dedication waiver (http://creativecommons.org/publicdomain/zero/1.0/)
applies to the data made available in this article, unless otherwise stated."""


_URLS = {
    _DATASETNAME: "https://github.com/cambridgeltl/bio-simverb/blob/master/wvlib/word-similarities/bio-simlex/Bio-SimLex.txt?raw=true",
}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class BioSimlexDataset(datasets.GeneratorBasedBuilder):
    """Bio-SimLex enables intrinsic evaluation of word representations. Config schema as source gives score between
    0-10 for pairs of words. Config schema as bigbio gives label 0 and 1 for pairs of words."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bio_simlex_source",
            version=SOURCE_VERSION,
            description="bio_simlex source schema",
            schema="source",
            subset_id="bio_simlex",
        ),
        BigBioConfig(
            name="bio_simlex_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="bio_simlex BigBio schema",
            schema="bigbio_pairs",
            subset_id="bio_simlex",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bio_simlex_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "score": datasets.Value("float"),
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

        url = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, line in enumerate(f):
                word1, word2, score = line.split("\t")
                if self.config.schema == "source":
                    yield id_, {
                        "text_1": word1,
                        "text_2": word2,
                        "score": score,
                    }

                elif self.config.schema == "bigbio_pairs":
                    yield id_, {
                        "id": str(id_),
                        "document_id": "NULL",
                        "text_1": word1,
                        "text_2": word2,
                        "label": "0" if float(score) < 5 else "1",
                    }


# This allows you to run your dataloader with `python bio_simlex.py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
