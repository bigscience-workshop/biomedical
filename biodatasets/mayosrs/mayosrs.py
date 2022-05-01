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
import requests
from typing import Dict, List, Tuple

import datasets
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

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

_HOMEPAGE = "https://nlp.cs.vcu.edu/data.html#mayosrs"

_LICENSE = "Unknown"

_URLS = {
    "mayosrs": [
        "https://nlp.cs.vcu.edu/data/similarity-data/MayoSRS.gold",
        "https://nlp.cs.vcu.edu/data/similarity-data/MayoSRS.terms",
    ]
}

_SUPPORTED_TASKS = [
    Tasks.SEMANTIC_SIMILARITY
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

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
        def custom_download(src_url: str, dst_path: str):
            cookies = {'NotificationDone': "true"}
            response = requests.get(src_url, cookies=cookies)
            with open(dst_path, "w") as handle:
                handle.write(response.text)
        data_dir = dl_manager.download_custom(urls, custom_download)

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
            code = filepath[0]
            texts = filepath[1]

            data_code = []
            with open(code, encoding="utf-8") as csv_file:
                csv_reader_code = csv.reader(
                    csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
                )
                for id_, row in enumerate(csv_reader_code):
                    label, code1, code2 = row[0].split("<>")
                    data_code.append([label, code1, code2])

            data_texts = []
            with open(texts, encoding="utf-8") as csv_file:
                csv_reader_texts = csv.reader(
                    csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
                )

                for id_, row in enumerate(csv_reader_texts):
                    text_1, text_2 = row[0].split("<>")
                    data_texts.append([text_1, text_2])

            data = []
            for i in range(len(data_code)):
                data.append(sum(list(zip(data_texts, data_code))[i], []))

            if self.config.schema == "source":
                for id_, row in enumerate(data):
                    text_1, text_2, label, code_1, code_2 = row

                    yield id_, {
                        "text_1": text_1,
                        "text_2": text_2,
                        "label": float(label),
                        "code_1": code_1,
                        "code_2": code_2,
                    }

            elif self.config.schema == "bigbio_pairs":
                uid = 0
                for id_, row in enumerate(data):
                    uid += 1
                    text_1, text_2, label, _, _ = row
                    yield id_, {
                        "id": uid,  # uid is an unique identifier for every record that starts from 1
                        "document_id": "NULL",
                        "text_1": text_1,
                        "text_2": text_2,
                        "label": str(label),
                    }

        else:
            print("There's no test/val split available for the given dataset")
            return
