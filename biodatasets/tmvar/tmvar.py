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


"""\
This dataset contains 500 PubMed articles manually annotated with mutation mentions of various kinds. It can be used for NER tasks
"""


from distutils.command.config import config
from multiprocessing.sharedctypes import Value
import os
from pydoc import doc
from typing import List, Tuple, Dict, Iterator

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks
import itertools

_CITATION = """\
@article{wei2018tmvar,
title={tmVar 2.0: integrating genomic variant information from literature with dbSNP and ClinVar for precision medicine},
author={Wei, Chih-Hsuan and Phan, Lon and Feltz, Juliana and Maiti, Rama and Hefferon, Tim and Lu, Zhiyong},
journal={Bioinformatics},
volume={34},
number={1},
pages={80--87},
year={2018},
publisher={Oxford University Press}
}
"""

_DATASETNAME = "tmvar"

_DESCRIPTION = """\
This dataset contains 500 PubMed articles manually annotated with mutation mentions of various kinds. It can be used for NER tasks
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/"

_LICENSE = "freely available"

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: {
        "v1": "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/tmVar/tmVarCorpus.zip",
        "v2": "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/tmVar/tmVar.Normalization.txt",
    }
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

logger = datasets.utils.logging.get_logger(__name__)


class tmvarDataset(datasets.GeneratorBasedBuilder):
    """
    The tmVar dataset contains 500 PubMed articles manually annotated with mutation
    mentions of various kinds.
    It can be used for biomedical NER tasks
    """

    DEFAULT_CONFIG_NAME = "tmvar_v1_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []
    for key in _URLS[_DATASETNAME].keys():
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"{_DATASETNAME}_{key}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} {key} source schema",
                schema="source",
                subset_id=f"{key}",
            )
        )
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"{_DATASETNAME}_{key}_bigbio_kb",
                version=BIGBIO_VERSION,
                description=f"{_DATASETNAME} {key} BigBio schema",
                schema="bigbio_kb",
                subset_id=f"{key}",
            )
        )

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`
        version = version = self.config.name.split("_")[1]

        if self.config.schema == "source":
            # Hacky fix for different fields in v1 and v2
            if version == "v1":
                features = datasets.Features(
                    {
                        "pmid": datasets.Value("string"),
                        "passages": [
                            {
                                "type": datasets.Value("string"),
                                "text": datasets.Sequence(datasets.Value("string")),
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                            }
                        ],
                        "entities": [
                            {
                                "text": datasets.Sequence(datasets.Value("string")),
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                                "concept_id": datasets.Value("string"),
                                "semantic_type_id": datasets.Sequence(
                                    datasets.Value("string")
                                ),
                            }
                        ],
                    }
                )
            elif version == "v2":
                features = datasets.Features(
                    {
                        "pmid": datasets.Value("string"),
                        "passages": [
                            {
                                "type": datasets.Value("string"),
                                "text": datasets.Sequence(datasets.Value("string")),
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                            }
                        ],
                        "entities": [
                            {
                                "text": datasets.Sequence(datasets.Value("string")),
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                                "concept_id": datasets.Value("string"),
                                "semantic_type_id": datasets.Sequence(
                                    datasets.Value("string")
                                ),
                                "rsid": datasets.Value("string"),
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
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        version = self.config.name.split("_")[1]
        # Hacky fix for inconsitent data format in tmvar
        # v1 is a zip of train and test files
        # v2 is a single .txt file with only the test data
        urls = _URLS[_DATASETNAME][version]
        if version == "v1":
            data_dir = dl_manager.download_and_extract(urls)
            train_filename = "tmVarCorpus\\train.PubTator.txt"
            test_filename = "tmVarCorpus\\test.PubTator.txt"
            train_filepath = os.path.join(data_dir, train_filename)
            test_filepath = os.path.join(data_dir, test_filename)
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
        elif version == "v2":
            data_dir = dl_manager.download(urls)
            test_filename = "tmVar.Normalization.txt"
            test_filepath = os.path.join(data_dir, test_filename)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": test_filepath,
                    },
                )
            ]

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # Hacky fix for inconsistent text format between v1 and v2
        # v1 is properly parsed by pubtator, but v2 fails
        version = self.config.name.split("_")[1]
        if self.config.schema == "source":
            with open(filepath, "r", encoding="utf8") as fstream:
                for raw_document in self.generate_raw_docs(fstream):
                    document = self.parse_raw_doc(raw_document)
                    for entity in document["entities"]:
                        entity.pop("rsid")
                    yield document["pmid"], document

        elif self.config.schema == "bigbio_kb":
            with open(filepath, "r", encoding="utf8") as fstream:
                for raw_document in self.generate_raw_docs(fstream):
                    document = self.parse_raw_doc(raw_document)
                    pmid = document.pop("pmid")
                    document["id"] = pmid
                    document["document_id"] = pmid
                    entities_ = []
                    for entity in document["entities"]:
                        if entity.get("rsid", ""):
                            normalized = [
                                {"db_name": "dbsnp", "db_id": entity.get("rsid")}
                            ]
                        else:
                            normalized = []
                        entities_.append(
                            {
                                "id": entity["concept_id"],
                                "type": entity["semantic_type_id"],
                                "text": entity["text"],
                                "normalized": normalized,
                                "offsets": entity["offsets"],
                            }
                        )
                    uid = itertools.count(0)
                    for passage in document["passages"]:
                        passage["id"] = next(uid)
                    document["entities"] = entities_
                    document["relations"] = []
                    document["events"] = []
                    document["coreferences"] = []
                    yield document["document_id"], document

    def generate_raw_docs(self, fstream):
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
            # Hacky fix for inconsistent formats between v1 and v2
            if len(mentions) == 6:
                (
                    pmid_,
                    start_idx,
                    end_idx,
                    mention,
                    semantic_type_id,
                    entity_id,
                ) = mentions
                rsid = None
            elif len(mentions) == 7:
                (
                    pmid_,
                    start_idx,
                    end_idx,
                    mention,
                    semantic_type_id,
                    entity_id,
                    rsid,
                ) = mentions
            else:
                logger.warning("Inconsistent entity format found. Skipping")
                logger.warning(f"Document ID: {pmid} Line: {line}")

            entity = {
                "offsets": [[int(start_idx), int(end_idx)]],
                "text": [mention],
                "semantic_type_id": semantic_type_id.split(","),
                "concept_id": entity_id,
                "rsid": rsid,
            }
            entities.append(entity)
        return {"pmid": pmid, "passages": passages, "entities": entities}
