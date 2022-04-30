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
The Chemical Exposure Information (CEI) Corpus consists of 3661 PubMed publication abstracts manually annotated by experts according to a taxonomy.
The taxonomy consists of 32 classes in a hierarchy. Zero or more class labels are assigned to each sentence in the corpus.
The labels are found under the "labels" directory, while the tokenized text can be found under "text" directory.
The filenames are the corresponding PubMed IDs (PMID).
"""

from pathlib import Path
import re
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks


_CITATION = """\
@article{,
  author    = {Larsson, Kristin and Baker, Simon and Silins, Ilona and Guo, Yufan and Stenius, Ulla and Korhonen, Anna and Berglund, Marika},
  title     = {Text mining for improved exposure assessment},
  journal   = {PloS one},
  volume    = {12},
  year      = {2017},
  url       = {https://doi.org/10.1371/journal.pone.0173132},
  doi       = {10.1371/journal.pone.0173132},
  biburl    = {https://journals.plos.org/plosone/article/citation/bibtex?id=10.1371/journal.pone.0173132},
  bibsource = {PloS one}
}
"""

_DATASETNAME = "cei"

_DESCRIPTION = """\
The Chemical Exposure Information (CEI) Corpus consists of 3661 PubMed publication abstracts manually annotated by experts according to a taxonomy.
The taxonomy consists of 32 classes in a hierarchy. Zero or more class labels are assigned to each sentence in the corpus.
The labels are found under the "labels" directory, while the tokenized text can be found under "text" directory.
The filenames are the corresponding PubMed IDs (PMID).
"""

_HOMEPAGE = "https://github.com/sb895/chemical-exposure-information-corpus"

_LICENSE = "GPL-3.0 License"

_URLS = {
    _DATASETNAME: "https://github.com/sb895/chemical-exposure-information-corpus/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

LABEL_REGEX = re.compile(r"[BE][a-z\-\ ]+")


class CieDataset(datasets.GeneratorBasedBuilder):
    """The Chemical Exposure Information (CEI) Corpus consists of 3661 PubMed publication abstracts manually annotated by experts according to a taxonomy.
    The taxonomy consists of 32 classes in a hierarchy. Zero or more class labels are assigned to each sentence in the corpus.
    The labels are found under the "labels" directory, while the tokenized text can be found under "text" directory.
    The filenames are the corresponding PubMed IDs (PMID)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        BigBioConfig(
            name=f"{_DATASETNAME}_bigbio_text",
            version=BIGBIO_VERSION,
            description=f"{_DATASETNAME} BigBio schema",
            schema="bigbio_text",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label_text": datasets.Value("string"),
                    "labels": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

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
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "base_dir": (
                        Path(data_dir) / "chemical-exposure-information-corpus-master"
                    ),
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, base_dir: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        text_files = list(base_dir.glob("./text/*.txt"))

        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            for text_file in text_files:
                key, example = self._read_example_from_file(text_file)
                yield key, example

        elif self.config.schema == "bigbio_[bigbio_schema_name]":
            # TODO: yield (key, example) tuples in the bigbio schema
            for text_file in text_files:
                key, example = self._read_example_from_file_in_kb_schema(text_file)
                yield key, example

    def _read_example_from_file(self, text_file: Path) -> Tuple[str, Dict]:
        label_file = text_file.parent.parent / "labels" / text_file.name
        with open(text_file, encoding="utf-8") as fp:
            text = fp.read().rstrip()
        with open(label_file, encoding="utf-8") as fp:
            label_text = fp.read()
        labels = [l.strip(" -") for l in LABEL_REGEX.findall(label_text)]
        key = text_file.name.rsplit(".", 1)[0]
        example = {
            "id": key,
            "document_id": key,
            "text": text,
            "label_text": label_text,
            "labels": labels,
        }
        return key, example

    def _read_example_from_file_in_kb_schema(self, text_file: Path) -> Tuple[str, Dict]:
        key, example = self._read_example_from_file(text_file)
        example = {
            k: v
            for k, v in example.items()
            if k in {"id", "document_id", "text", "labels"}
        }
        return key, example


if __name__ == "__main__":
    datasets.load_dataset(__file__)
