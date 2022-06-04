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
from pathlib import Path

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{DBLP:journals/bioinformatics/BakerSGAHSK16,
  author    = {Simon Baker and
               Ilona Silins and
               Yufan Guo and
               Imran Ali and
               Johan H{\"{o}}gberg and
               Ulla Stenius and
               Anna Korhonen},
  title     = {Automatic semantic classification of scientific literature
               according to the hallmarks of cancer},
  journal   = {Bioinform.},
  volume    = {32},
  number    = {3},
  pages     = {432--440},
  year      = {2016},
  url       = {https://doi.org/10.1093/bioinformatics/btv585},
  doi       = {10.1093/bioinformatics/btv585},
  timestamp = {Thu, 14 Oct 2021 08:57:44 +0200},
  biburl    = {https://dblp.org/rec/journals/bioinformatics/BakerSGAHSK16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "hallmarks_of_cancer"

_DESCRIPTION = """\
The Hallmarks of Cancer (HOC) Corpus consists of 1852 PubMed publication
abstracts manually annotated by experts according to a taxonomy. The taxonomy
consists of 37 classes in a hierarchy. Zero or more class labels are assigned
to each sentence in the corpus. The labels are found under the "labels"
directory, while the tokenized text can be found under "text" directory.
The filenames are the corresponding PubMed IDs (PMID).
"""

_HOMEPAGE = "https://github.com/sb895/Hallmarks-of-Cancer"

_LICENSE = Licenses.GPL_3p0

_URLs = {
    "corpus": "https://github.com/sb895/Hallmarks-of-Cancer/archive/refs/heads/master.zip",
    "split_indices": "https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz"    
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CLASS_NAMES = [
    'evading growth suppressors',
    'tumor promoting inflammation',
    'enabling replicative immortality',
    'cellular energetics',
    'resisting cell death',
    'activating invasion and metastasis',
    'genomic instability and mutation',
    'none',
    'inducing angiogenesis',
    'sustaining proliferative signaling',
    'avoiding immune destruction'
]


class HallmarksOfCancerDataset(datasets.GeneratorBasedBuilder):
    """Hallmarks Of Cancer Dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="hallmarks_of_cancer_source",
            version=SOURCE_VERSION,
            description="Hallmarks of Cancer source schema",
            schema="source",
            subset_id="hallmarks_of_cancer",
        ),
        BigBioConfig(
            name="hallmarks_of_cancer_bigbio_text",
            version=BIGBIO_VERSION,
            description="Hallmarks of Cancer Biomedical schema",
            schema="bigbio_text",
            subset_id="hallmarks_of_cancer",
        ),
    ]
    DEFAULT_CONFIG_NAME = "hallmarks_of_cancer_source"

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": [datasets.ClassLabel(names=_CLASS_NAMES)],
                }
            )

        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "corpuspath": Path(data_dir["corpus"]),
                    "indicespath": Path(data_dir["split_indices"]) / "data_generation/indexing/HoC/train_pmid.tsv"                
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "corpuspath": Path(data_dir["corpus"]),
                    "indicespath": Path(data_dir["split_indices"]) / "data_generation/indexing/HoC/test_pmid.tsv"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "corpuspath": Path(data_dir["corpus"]),
                    "indicespath": Path(data_dir["split_indices"]) / "data_generation/indexing/HoC/dev_pmid.tsv"              
                },
            ),
        ]

    def _generate_examples(self, corpuspath: Path, indicespath: Path):

        indices = indicespath.read_text(encoding="utf8").strip("\n").split(",")
        dataset_dir = corpuspath / "Hallmarks-of-Cancer-master"
        texts_dir = dataset_dir / "text"
        labels_dir = dataset_dir / "labels"

        uid = 1
        for document_index, document in enumerate(indices):
            text_file = texts_dir / document
            label_file = labels_dir / document
            text = text_file.read_text(encoding="utf8").strip("\n")
            labels = label_file.read_text(encoding="utf8").strip("\n")

            sentences = text.split("\n")
            labels = labels.split("<")[1:]

            for example_index, example_pair in enumerate(zip(sentences, labels)):
                sentence, label = example_pair

                label = label.strip()
                
                if label == "":
                    label = "none"

                multi_labels = [m_label.strip() for m_label in label.split("AND")]
                unique_multi_labels = {
                    m_label.split("--")[0].lower().lstrip() for m_label in multi_labels if m_label != "NULL"
                }

                arrow_file_unique_key = 100 * document_index + example_index
                if self.config.schema == "source":
                    yield arrow_file_unique_key, {
                        "document_id": f"{text_file.name.split('.')[0]}_{example_index}",
                        "text": sentence,
                        "label": list(unique_multi_labels),
                    }
                elif self.config.schema == "bigbio_text":
                    yield arrow_file_unique_key, {
                        "id": uid,
                        "document_id": f"{text_file.name.split('.')[0]}_{example_index}",
                        "text": sentence,
                        "labels": list(unique_multi_labels),
                    }
                    uid += 1
