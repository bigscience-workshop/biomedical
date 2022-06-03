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


import itertools
import os
from pydoc import doc
from typing import Dict, Iterator, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{wei2013tmvar,
  title={tmVar: a text mining approach for extracting sequence variants in biomedical literature},
  author={Wei, Chih-Hsuan and Harris, Bethany R and Kao, Hung-Yu and Lu, Zhiyong},
  journal={Bioinformatics},
  volume={29},
  number={11},
  pages={1433--1439},
  year={2013},
  publisher={Oxford University Press}
}
"""

_DATASETNAME = "tmvar_v1"

_DESCRIPTION = """This dataset contains 500 PubMed articles manually annotated with mutation mentions of various kinds. It can be used for NER tasks only.
The dataset is split into train(334) and test(166) splits"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/tmVar/tmVarCorpus.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

logger = datasets.utils.logging.get_logger(__name__)


class TmvarV1Dataset(datasets.GeneratorBasedBuilder):
    """
    The tmVar dataset contains 500 PubMed articles manually annotated with mutation
    mentions of various kinds.
    It can be used for biomedical NER tasks
    """

    DEFAULT_CONFIG_NAME = "tmvar_v1_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []
    BUILDER_CONFIGS.append(
        BigBioConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        )
    )
    BUILDER_CONFIGS.append(
        BigBioConfig(
            name=f"{_DATASETNAME}_bigbio_kb",
            version=BIGBIO_VERSION,
            description=f"{_DATASETNAME} BigBio schema",
            schema="bigbio_kb",
            subset_id=f"{_DATASETNAME}",
        )
    )

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "passages": [
                        {
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": [datasets.Value("int32")],
                        }
                    ],
                    "entities": [
                        {
                            "text": datasets.Value("string"),
                            "offsets": [datasets.Value("int32")],
                            "concept_id": datasets.Value("string"),
                            "semantic_type_id": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        url = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(url)
        train_filepath = os.path.join(data_dir, "tmVarCorpus", "train.PubTator.txt")
        test_filepath = os.path.join(data_dir, "tmVarCorpus", "test.PubTator.txt")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_filepath,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_filepath,
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            with open(filepath, "r", encoding="utf8") as fstream:
                for raw_document in self.generate_raw_docs(fstream):
                    document = self.parse_raw_doc(raw_document)
                    yield document["pmid"], document

        elif self.config.schema == "bigbio_kb":
            with open(filepath, "r", encoding="utf8") as fstream:
                uid = itertools.count(0)
                for raw_document in self.generate_raw_docs(fstream):
                    document = self.parse_raw_doc(raw_document)
                    pmid = document.pop("pmid")
                    document["id"] = next(uid)
                    document["document_id"] = pmid

                    entities_ = []
                    for entity in document["entities"]:
                        entities_.append(
                            {
                                "id": next(uid),
                                "type": entity["semantic_type_id"],
                                "text": [entity["text"]],
                                "normalized": [],
                                "offsets": [entity["offsets"]],
                            }
                        )
                    for passage in document["passages"]:
                        passage["id"] = next(uid)

                    document["entities"] = entities_
                    document["relations"] = []
                    document["events"] = []
                    document["coreferences"] = []
                    yield document["document_id"], document

    def generate_raw_docs(self, fstream):
        """
        Given a filestream, this function yields documents from it
        """
        raw_document = []
        for line in fstream:
            if line.strip():
                raw_document.append(line.strip())
            elif raw_document:
                yield raw_document
                raw_document = []
        if raw_document:
            yield raw_document

    def parse_raw_doc(self, raw_doc):
        pmid, _, title = raw_doc[0].split("|")
        pmid = int(pmid)
        _, _, abstract = raw_doc[1].split("|")

        if self.config.schema == "source":
            passages = [
                {"type": "title", "text": title, "offsets": [0, len(title)]},
                {
                    "type": "abstract",
                    "text": abstract,
                    "offsets": [len(title) + 1, len(title) + len(abstract) + 1],
                },
            ]
        elif self.config.schema == "bigbio_kb":
            passages = [
                {"type": "title", "text": [title], "offsets": [[0, len(title)]]},
                {
                    "type": "abstract",
                    "text": [abstract],
                    "offsets": [[len(title) + 1, len(title) + len(abstract) + 1]],
                },
            ]

        entities = []
        for line in raw_doc[2:]:
            mentions = line.split("\t")
            (
                pmid_,
                start_idx,
                end_idx,
                mention,
                semantic_type_id,
                entity_id,
            ) = mentions

            entity = {
                "offsets": [int(start_idx), int(end_idx)],
                "text": mention,
                "semantic_type_id": semantic_type_id,
                "concept_id": entity_id,
            }
            entities.append(entity)

        return {"pmid": pmid, "passages": passages, "entities": entities}
