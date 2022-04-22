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
This repository contains the evaluation datasets for the paper Bio-SimVerb and
Bio-SimLex: Wide-coverage Evaluation Sets of Word Similarity in Biomedicine by
Billy Chiu, Sampo Pyysalo and Anna Korhonen.
"""

import csv
from typing import Dict, List, Tuple

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{chiu2018bio,
  title={Bio-SimVerb and Bio-SimLex: wide-coverage evaluation sets of word similarity in biomedicine},
  author={Chiu, Billy and Pyysalo, Sampo and Vuli{\'c}, Ivan and Korhonen, Anna},
  journal={BMC bioinformatics},
  volume={19},
  number={1},
  pages={1--13},
  year={2018},
  publisher={BioMed Central}
}
"""

_DATASETNAME = "bio_sim_verb"


_DESCRIPTION = """
This repository contains the evaluation datasets for the paper
Bio-SimVerb and Bio-SimLex: Wide-coverage Evaluation Sets of Word
Similarity in Biomedicine by Billy Chiu, Sampo Pyysalo and Anna Korhonen.
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

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/cambridgeltl/bio-simverb/master/wvlib/word-similarities/bio-simverb/Bio-SimVerb.txt",
}

_SUPPORTED_TASKS = [
    Tasks.SEMANTIC_SIMILARITY
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
#  Append "Dataset" to the class name: BioASQ --> BioasqDataset
class BioSimVerb(datasets.GeneratorBasedBuilder):
    """Evaluation of word similarity in biomedical texts."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bio_sim_verb_source",
            version=SOURCE_VERSION,
            description="bio_sim_verb source schema",
            schema="source",
            subset_id="bio_sim_verb",
        ),
        BigBioConfig(
            name="bio_sim_verb_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="bio_sim_verb BigBio schema",
            schema="bigbio_pairs",
            subset_id="bio_sim_verb",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bio_sim_verb_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "label": datasets.Value("float32"),
                }
            )

        # Using in pairs schema
        elif self.config.schema == "bigbio_pairs":
            features = schemas.pairs_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dir, "split": "train"},
            )
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )

            if self.config.schema == "source":
                for id_, row in enumerate(csv_reader):
                    text_1, text_2, label = row
                    yield id_, {
                        "text_1": text_1,
                        "text_2": text_2,
                        "label": float(label),
                    }

            elif self.config.schema == "bigbio_pairs":
                uid = 0
                for id_, row in enumerate(csv_reader):
                    uid += 1
                    text_1, text_2, label = row
                    yield id_, {
                        "id": str(uid),
                        "document_id": "NULL",
                        "text_1": text_1,
                        "text_2": text_2,
                        "label": str(label),
                    }
