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
This dataset contains 500 PubMed articles manually annotated with mutation
mentions of various kinds and dbsnp normalizations for each of them.  In
addition, it contains variant normalization options such as allele-specific
identifiers from the ClinGen Allele Registry It can be used for NER tasks and
NED tasks, This dataset does NOT have splits.
"""
import itertools

import datasets
from bioc import pubtator

from .bigbiohub import BigBioConfig, Tasks, kb_features

_CITATION = """\
@article{wei2022tmvar,
  title={tmVar 3.0: an improved variant concept recognition and normalization tool},
  author={Wei, Chih-Hsuan and Allot, Alexis and Riehle, Kevin and Milosavljevic, Aleksandar and Lu, Zhiyong},
  journal={Bioinformatics},
  volume={38},
  number={18},
  pages={4449--4451},
  year={2022},
  publisher={Oxford University Press}
}
"""
_LANGUAGES = ["English"]
_PUBMED = True
_LOCAL = False

_DATASETNAME = "tmvar_v3"
_DISPLAYNAME = "tmVar v3"

_DESCRIPTION = """\
This dataset contains 500 PubMed articles manually annotated with mutation \
mentions of various kinds and dbsnp normalizations for each of them.  In \
addition, it contains variant normalization options such as allele-specific \
identifiers from the ClinGen Allele Registry It can be used for NER tasks and \
NED tasks, This dataset does NOT have splits.
"""

_HOMEPAGE = "https://github.com/ncbi/tmVar3"

_LICENSE = "UNKNOWN"

_URLS = {_DATASETNAME: "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/tmVar3/tmVar3Corpus.txt"}
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]
_SOURCE_VERSION = "3.0.0"
_BIGBIO_VERSION = "1.0.0"
logger = datasets.utils.logging.get_logger(__name__)


class TmvarV3Dataset(datasets.GeneratorBasedBuilder):
    """
    This dataset contains 500 PubMed articles manually annotated with mutation
    mentions of various kinds and various normalizations for each of them.
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
            name=f"{_DATASETNAME}_source_fixed",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema with fixed offsets",
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
                            "semantic_type_id": datasets.Value("string"),
                            "normalized": {
                                key: datasets.Sequence(datasets.Value("string"))
                                for key in type_to_db_mapping.keys()
                            },
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = kb_features
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
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

    def get_normalizations(self, id, type, doc_id):
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
                    db_ids = db_id.split(",")
                    base_dict[db_name].extend(db_ids)
                else:
                    logger.info(
                        f"Malformed normalization in Document {doc_id}. Type: {type}, Number: {id}"
                    )
                    continue
        return base_dict

    def _correct_wrong_offsets(self, entities, pmid):
        """
        Offsets in the document 21904390 is wrong. Correct them manually.
        """
        wrong_offsets = {
            "21904390": {
                (343, 347): [342, 346],
                (753, 757): [751, 755],
                (1156, 1160): [1153, 1157],
                (1487, 1491): [1483, 1487],
                (1631, 1635): [1627, 1631],
                (1645, 1659): [1640, 1654],
                (2043, 2047): [2037, 2041],
            }
        }
        if pmid in wrong_offsets:
            for entity in entities:
                if (entity["offsets"][0][0], entity["offsets"][0][1]) in wrong_offsets[
                    pmid
                ]:
                    entity["offsets"][0] = wrong_offsets[pmid][
                        (entity["offsets"][0][0], entity["offsets"][0][1])
                    ]
        return entities

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
                        "semantic_type_id": mention.type,
                        "normalized": self.get_normalizations(
                            mention.id,
                            mention.type,
                            doc.pmid,
                        ),
                    }
                    for mention in doc.annotations
                ]

                if "_fixed" in self.config.name:
                    document["entities"] = self._correct_wrong_offsets(
                        document["entities"], doc.pmid
                    )
                    
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
                        "type": mention.type,
                        "normalized": self.get_normalizations(
                            mention.id, mention.type, doc.pmid
                        ),
                    }
                    for mention in doc.annotations
                ]
                document["entities"] = self._correct_wrong_offsets(
                    document["entities"], doc.pmid
                )
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

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            for source_example in self.pubtator_to_source(filepath):
                yield source_example["pmid"], source_example
        elif self.config.schema == "bigbio_kb":
            for bigbio_example in self.pubtator_to_bigbio_kb(filepath):
                yield bigbio_example["document_id"], bigbio_example
