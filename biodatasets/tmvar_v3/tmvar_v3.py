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


"""\This dataset contains 500 PubMed articles manually annotated with mutation mentions of various kinds and dbsnp normalizations for each of them. 
In addition, it contains variant normalization options such as allele-specific identifiers from the ClinGen Allele Registry
It can be used for NER tasks and NED tasks, This dataset does NOT have splits"""


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
from bioc import pubtator

_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2204.03637,
  doi = {10.48550/ARXIV.2204.03637},
  
  url = {https://arxiv.org/abs/2204.03637},
  
  author = {Wei, Chih-Hsuan and Allot, Alexis and Riehle, Kevin and Milosavljevic, Aleksandar and Lu, Zhiyong},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {tmVar 3.0: an improved variant concept recognition and normalization tool},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

"""

_DATASETNAME = "tmvar_v3"

_DESCRIPTION = """This dataset contains 500 PubMed articles manually annotated with mutation mentions of various kinds and dbsnp normalizations for each of them. 
In addition, it contains variant normalization options such as allele-specific identifiers from the ClinGen Allele Registry
It can be used for NER tasks and NED tasks, This dataset does NOT have splits"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/"

_LICENSE = "freely available"

_URLS = {_DATASETNAME: "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/tmVar3/tmVar3Corpus.txt"}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "3.0.0"

_BIGBIO_VERSION = "1.0.0"

logger = datasets.utils.logging.get_logger(__name__)


class TmvarV3Dataset(datasets.GeneratorBasedBuilder):
    """
    This dataset contains 500 PubMed articles manually annotated with mutation mentions of various kinds and various normalizations for each of them.
    """

    DEFAULT_CONFIG_NAME = "tmvar_v3_source"
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

        type_to_db_mapping = {
            "CorrespondingGene": "NCBI Gene",
            "tmVar": "tmVar",
            "dbSNP": "dbSNP",
            "VariantGroup": "VariantGroup",
            "NCBI Taxonomy": "NCBI Taxonomy",
        }
        if self.config.schema == "source":
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
                            "semantic_type_id": datasets.Sequence(
                                datasets.Value("string")
                            ),
                            "normalized": {
                                key: datasets.Sequence(datasets.Value("string"))
                                for key in type_to_db_mapping.keys()
                            },
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

        url = _URLS[_DATASETNAME]
        test_filepath = dl_manager.download(url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_filepath,
                },
            )
        ]

    def get_normalizations(self, id, type):
        """
        Given a type and a number of normalizations ids, this function returns a dictionary of the normalized ids
        """
        base_dict = {
            key: []
            for key in [
                "tmVar",
                "CorrespondingGene",
                "dbSNP",
                "VariantGroup",
                "NCBI Taxonomy",
            ]
        }
        ids = id.split(";")
        if type in ["CellLine", "Species"]:
            id_vals = ids[0].split(",")
            base_dict["NCBI Taxonomy"] = id_vals

        elif type == "Gene":
            id_vals = ids[0].split(",")
            base_dict["CorrespondingGene"] = id_vals

        else:
            for id in ids:
                if "|" in id:
                    base_dict["tmVar"].append(id)
                elif id[:2] == "rs":
                    base_dict["dbSNP"].append(id[2:])
                elif ":" in id:
                    db_name, db_id = id.split(":")
                    if db_name == "RS#":
                        db_name = "dbSNP"
                    # Hacky fix below for doc ID: 18272172
                    elif db_name == "Va1iantGroup":
                        db_name = "VariantGroup"
                    elif db_name == "Gene":
                        db_name = "CorrespondingGene"
                    elif db_name == "Disease":
                        continue
                    base_dict[db_name].append(db_id)
                else:
                    logger.warn(f"Malformed normalization. Type: {type}, Number: {id}")
                    continue
        return base_dict

    def pubtator_to_source(self, filepath):
        """
        Converts pubtator to source schema
        """
        with open(filepath, "r", encoding="utf8") as fstream:
            for doc in pubtator.iterparse(fstream):
                document = {}
                document["pmid"] = doc.pmid
                title = doc.title
                abstract = doc.abstract
                document["passages"] = [
                    {"type": "title", "text": [title], "offsets": [[0, len(title)]]},
                    {
                        "type": "abstract",
                        "text": [abstract],
                        "offsets": [[len(title) + 1, len(title) + len(abstract) + 1]],
                    },
                ]
                document["entities"] = [
                    {
                        "offsets": [[mention.start, mention.end]],
                        "text": [mention.text],
                        "semantic_type_id": [mention.type],
                        "normalized": self.get_normalizations(mention.id, mention.type),
                    }
                    for mention in doc.annotations
                ]
                yield document

    def pubtator_to_bigbio_kb(self, filepath):
        """
        Converts pubtator to bigbio_kb schema
        """
        with open(filepath, "r", encoding="utf8") as fstream:
            uid = itertools.count(0)
            for doc in pubtator.iterparse(fstream):
                document = {}
                title = doc.title
                abstract = doc.abstract
                document["id"] = next(uid)
                document["document_id"] = doc.pmid
                document["passages"] = [
                    {
                        "id": next(uid),
                        "type": "title",
                        "text": [title],
                        "offsets": [[0, len(title)]],
                    },
                    {
                        "id": next(uid),
                        "type": "abstract",
                        "text": [abstract],
                        "offsets": [[len(title) + 1, len(title) + len(abstract) + 1]],
                    },
                ]
                document["entities"] = [
                    {
                        "id": next(uid),
                        "offsets": [[mention.start, mention.end]],
                        "text": [mention.text],
                        "type": [mention.type],
                        "normalized": self.get_normalizations(mention.id, mention.type),
                    }
                    for mention in doc.annotations
                ]
                db_id_mapping = {
                    "dbSNP": "dbSNP",
                    "CorrespondingGene": "NCBI Gene",
                    "tmVar": "dbSNP",
                }
                for entity in document["entities"]:
                    normalized_bigbio_kb = []
                    for key, id_list in entity["normalized"].items():
                        if key in db_id_mapping.keys():
                            normalized_bigbio_kb.extend(
                                [
                                    {"db_name": db_id_mapping[key], "db_id": id}
                                    for id in id_list
                                ]
                            )
                    entity["normalized"] = normalized_bigbio_kb
                document["relations"] = []
                document["events"] = []
                document["coreferences"] = []
                yield document

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            for source_example in self.pubtator_to_source(filepath):
                yield source_example["pmid"], source_example
        elif self.config.schema == "bigbio_kb":
            for bigbio_example in self.pubtator_to_bigbio_kb(filepath):
                yield bigbio_example["document_id"], bigbio_example
