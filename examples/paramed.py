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
NEJM is a Chinese-English parallel corpus crawled from the New England Journal of Medicine website.
English articles are distributed through https://www.nejm.org/ and Chinese articles are distributed through
http://nejmqianyan.cn/. The corpus contains all article pairs (around 2000 pairs) since 2011.
The script loads dataset in bigbio schema (using schemas/text-to-text) AND/OR source (default) schema
"""
import os  # useful for paths
from typing import Dict, Iterable, List
from dataclasses import dataclass
import datasets

logger = datasets.logging.get_logger(__name__)



_CITATION = """\
@article{,
  author    = {Liu, Boxiang and Huang, Liang},
  title     = {ParaMed: a parallel corpus for English–Chinese translation in the biomedical domain},
  journal   = {BMC Medical Informatics and Decision Making},
  volume    = {21},
  year      = {2021},
  url       = {https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01621-8},
  doi       = {10.1186/s12911-021-01621-8}
}
"""
_DATASETNAME = "paramed"

_DESCRIPTION = """\
NEJM is a Chinese-English parallel corpus crawled from the New England Journal of Medicine website. 
English articles are distributed through https://www.nejm.org/ and Chinese articles are distributed through 
http://nejmqianyan.cn/. The corpus contains all article pairs (around 2000 pairs) since 2011.
"""

_HOMEPAGE = "https://github.com/boxiangliu/ParaMed"

_LICENSE = "Creative Commons Attribution 4.0 International"

_URLs = {
    "source": "https://github.com/boxiangliu/ParaMed/blob/master/data/nejm-open-access.tar.gz?raw=true",
    "bigbio_text_to_text": "https://github.com/boxiangliu/ParaMed/blob/master/data/nejm-open-access.tar.gz?raw=true",
}
_SUPPORTED_TASKS = ["TRANSLATION"]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_DATA_DIR = "./processed_data/open_access/open_access"

@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""
    name: str = None
    version: str = None
    description: str = None
    schema: str = None
    subset_id: str = None

class ParamedDataset(datasets.GeneratorBasedBuilder):
    """Write a short docstring documenting what this dataset is"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="paramed_source",
            version=SOURCE_VERSION,
            description="Paramed source schema",
            schema="source",
            subset_id="paramed",
        ),
        BigBioConfig(
            name="paramed_bigbio_text_to_text",
            version=BIGBIO_VERSION,
            description="Paramed BigBio schema",
            schema="bigbio_text_to_text",
            subset_id="paramed",
        )
    ]

    DEFAULT_CONFIG_NAME = "paramed_source"

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "text_1_name": datasets.Value("string"),
                    "text_2_name": datasets.Value("string")
                }
            )

        elif self.config.schema == "bigbio_text_to_text":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "text_1_name": datasets.Value("string"),
                    "text_2_name": datasets.Value("string")
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        my_urls = _URLs[self.config.schema]
        data_dir = os.path.join(dl_manager.download_and_extract(my_urls), _DATA_DIR)
        print(data_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "zh_file": os.path.join(data_dir, "nejm.train.zh"),
                    "en_file": os.path.join(data_dir, "nejm.train.en"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir,
                    "zh_file": os.path.join(data_dir, "nejm.dev.zh"),
                    "en_file": os.path.join(data_dir, "nejm.dev.en"),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir,
                    "zh_file": os.path.join(data_dir, "nejm.test.zh"),
                    "en_file": os.path.join(data_dir, "nejm.test.en"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self,
                           filepath,
                           zh_file,
                           en_file,
                           split):

        logger.info("generating examples from = %s", filepath)
        zh_file = open(zh_file, "r")
        en_file = open(en_file, "r")
        zh_file.seek(0)
        en_file.seek(0)
        zh_lines = zh_file.readlines()
        en_lines = en_file.readlines()

        assert len(en_lines) == len(zh_lines), "Line mismatch"

        if self.config.schema == "source":
            for key, (zh_line, en_line) in enumerate(zip(zh_lines, en_lines)):
                yield key, {
                    "document_id": str(key),
                    "text_1": zh_line,
                    "text_2": en_line,
                    "text_1_name": "zh",
                    "text_2_name": "en",
                }
            zh_file.close()
            en_file.close()

        elif self.config.schema == "bigbio_text_to_text":
            uid = 0
            for key, (zh_line, en_line) in enumerate(zip(zh_lines, en_lines)):
                uid += 1
                yield key, {
                    "id": str(uid),
                    "document_id": str(key),
                    "text_1": zh_line,
                    "text_2": en_line,
                    "text_1_name": "zh",
                    "text_2_name": "en",
                }
            zh_file.close()
            en_file.close()
