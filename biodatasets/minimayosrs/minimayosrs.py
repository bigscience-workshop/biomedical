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

import csv
from typing import Dict, List, Tuple

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

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

_HOMEPAGE = "https://nlp.cs.vcu.edu/data.html#minimayosrs"

_LICENSE = "Unknown"

_URLS = {
    "source": [
        "https://nlp.cs.vcu.edu/data/similarity-data/MiniMayoSRS.terms",
        "https://nlp.cs.vcu.edu/data/similarity-data/MiniMayoSRS.physicians",
        "https://nlp.cs.vcu.edu/data/similarity-data/MiniMayoSRS.coders",
    ],
    "bigbio_pairs": [
        "https://nlp.cs.vcu.edu/data/similarity-data/MiniMayoSRS.terms",
        "https://nlp.cs.vcu.edu/data/similarity-data/MiniMayoSRS.physicians",
        "https://nlp.cs.vcu.edu/data/similarity-data/MiniMayoSRS.coders",
    ],
}

_SUPPORTED_TASKS = [
    Tasks.SEMANTIC_SIMILARITY
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

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
                    "label_physicians": datasets.Value("float32"),
                    "code_1_physicians": datasets.Value("string"),
                    "code_2_physicians": datasets.Value("string"),
                    "label_coders": datasets.Value("float32"),
                    "code_1_coders": datasets.Value("string"),
                    "code_2_coders": datasets.Value("string"),
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

        urls = _URLS[self.config.schema]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if split == "train":
            # texts, physicians, coders
            texts = filepath[0]
            physicians = filepath[1]
            coders = filepath[2]

            data_texts = []
            with open(texts, encoding="utf-8") as csv_file:
                csv_reader_texts = csv.reader(
                    csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
                )
                for id_, row in enumerate(csv_reader_texts):
                    text_1, text_2 = row[0].split("<>")
                    data_texts.append([text_1, text_2])

            data_physicians = []
            with open(physicians, encoding="utf-8") as csv_file:
                csv_reader_physicians = csv.reader(
                    csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
                )

                for id_, row in enumerate(csv_reader_physicians):
                    label_physicians, code_1_physicians, code_2_physicians = row[0].split("<>")
                    data_physicians.append([label_physicians, code_1_physicians, code_2_physicians])

            data_coders = []
            with open(coders, encoding="utf-8") as csv_file:
                csv_reader_coders = csv.reader(
                    csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
                )

                for id_, row in enumerate(csv_reader_coders):
                    label_coders, code_1_coders, code_2_coders = row[0].split("<>")
                    data_coders.append([label_coders, code_1_coders, code_2_coders])

            data = []
            for i in range(len(data_coders)):
                data.append(sum(list(zip(data_texts, data_physicians, data_coders))[i], []))

            if self.config.schema == "source":
                for id_, row in enumerate(data):
                    (
                        text_1,
                        text_2,
                        label_physicians,
                        code_1_physicians,
                        code_2_physicians,
                        label_coders,
                        code_1_coders,
                        code_2_coders,
                    ) = row

                    yield id_, {
                        "text_1": text_1,
                        "text_2": text_2,
                        "label_physicians": float(label_physicians),
                        "code_1_physicians": code_1_physicians,
                        "code_2_physicians": code_2_physicians,
                        "label_coders": float(label_coders),
                        "code_1_coders": code_1_coders,
                        "code_2_coders": code_2_coders,
                    }

            elif self.config.schema == "bigbio_pairs":
                uid = 0
                for id_, row in enumerate(data):
                    uid += 1
                    text_1, text_2, label_physicians, _, _, _, _, _ = row
                    yield id_, {
                        "id": uid,  # uid is an unique identifier for every record that starts from 1
                        "document_id": "NULL",
                        "text_1": text_1,
                        "text_2": text_2,
                        "label": str(label_physicians),
                    }

        else:
            print("There's no test/val split available for the given dataset")
            return
