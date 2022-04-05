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
PubTator Central (PTC, https://www.ncbi.nlm.nih.gov/research/pubtator/) [1] is a web service for
exploring and retrieving bioconcept annotations in full text biomedical articles. PTC provides
automated annotations from state-of-the-art text mining systems for genes/proteins, genetic
variants, diseases, chemicals, species and cell lines, all available for immediate download. PTC
annotates PubMed (30 million abstracts), the PMC Open Access Subset and the Author Manuscript
Collection (3 million full text articles). Updated entity identification methods and a
disambiguation module [2] based on cutting-edge deep learning techniques provide increased accuracy.
This FTP repository aggregated all the bio-entity annotations in PTC in tab-separated text format.
The files are expected to be updated monthly.

REFERENCE:
---------------------------------------------------------------------------
[1] Wei C-H, Allot A, Leaman R and Lu Z (2019) "PubTator Central: Automated Concept Annotation for
    Biomedical Full Text Articles", Nucleic Acids Res.
[2] wei C-H, et al., (2019) "Biomedical Mention Disambiguation Using a Deep Learning Approach",
    ACM-BCB 2019, September 7-10, 2019, Niagara Falls, NY, USA.
[3] Wei C-H, Kao H-Y, Lu Z (2015) "GNormPlus: An Integrative Approach for Tagging Gene, Gene Family
    and Protein Domain", 2015, Article ID 918710
[4] Leaman R and Lu Z (2013) "TaggerOne: joint named entity recognition and normalization with
    semi-Markov Models", Bioinformatics, 32(18): 839-2846
[5] Wei C-H, Kao H-Y, Lu Z (2012) "SR4GN: a species recognition software tool for gene normalization",
    PLoS ONE,7(6):e38460
[6] Wei C-H, et al., (2017) "Integrating genomic variant information from literature with dbSNP and
    ClinVar for precision medicine", Bioinformatics,34(1): 80-87
"""

import re
from typing import Dict, Iterator, List, Tuple

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{10.1093/nar/gkz389,
  title        = {{PubTator central: automated concept annotation for biomedical full text articles}},
  author       = {Wei, Chih-Hsuan and Allot, Alexis and Leaman, Robert and Lu, Zhiyong},
  year         = 2019,
  month        = {05},
  journal      = {Nucleic Acids Research},
  volume       = 47,
  number       = {W1},
  pages        = {W587-W593},
  doi          = {10.1093/nar/gkz389},
  issn         = {0305-1048},
  url          = {https://doi.org/10.1093/nar/gkz389},
  eprint       = {https://academic.oup.com/nar/article-pdf/47/W1/W587/28880193/gkz389.pdf}
}
"""

_DATASETNAME = "pubtator_central"

_DESCRIPTION = """\
PubTator Central (PTC, https://www.ncbi.nlm.nih.gov/research/pubtator/) is a web service for
exploring and retrieving bioconcept annotations in full text biomedical articles. PTC provides
automated annotations from state-of-the-art text mining systems for genes/proteins, genetic
variants, diseases, chemicals, species and cell lines, all available for immediate download. PTC
annotates PubMed (30 million abstracts), the PMC Open Access Subset and the Author Manuscript
Collection (3 million full text articles). Updated entity identification methods and a
disambiguation module based on cutting-edge deep learning techniques provide increased accuracy.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/research/pubtator/"

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = ""

_URLS = {
    _DATASETNAME: "https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.gz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "2022.01.08"

_BIGBIO_VERSION = "1.0.0"


class PubtatorCentralDataset(datasets.GeneratorBasedBuilder):
    """PubTator Central"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pubtator_central_source",
            version=SOURCE_VERSION,
            description="PubTator Central source schema",
            schema="source",
            subset_id="pubtator_central",
        ),
        BigBioConfig(
            name="pubtator_central_bigbio_kb",
            version=BIGBIO_VERSION,
            description="PubTator Central BigBio schema",
            schema="bigbio_kb",
            subset_id="pubtator_central",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pubtator_central_source"

    # Maps the entity types in PubTator to the name of the database they are grounded to
    PUBTATOR_TYPE_TO_UNIFIED_DB_NAME = {
        "Gene": "ncbi_gene",
        "Chemical": "mesh",
        "Species": "ncbi_taxon",
        "Disease": "mesh",
        "Mutation": "ncbi_dbsnp",
        "CellLine": "cellosaurus",
        # TODO: I don't know what this is being grounded to. It's not documented anywhere AFAICT.
        "ProteinMutation": "-1",
    }

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "mentions": [
                        {
                            "concept_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": datasets.Sequence(datasets.Value("int32")),
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
        urls = _URLS[_DATASETNAME]
        filepath = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: str, split: str) -> Iterator[Tuple[str, Dict]]:
        if self.config.schema == "source":
            for pubtator_example in self._parse_pubtator_file(filepath):
                yield pubtator_example["pmid"], pubtator_example

        elif self.config.schema == "bigbio_kb":
            for pubtator_example in self._parse_pubtator_file(filepath):
                kb_example = self._pubtator_parse_to_bigbio_kb(pubtator_example)
                yield kb_example["id"], kb_example

    @staticmethod
    def _parse_pubtator_file(filepath: str) -> Iterator[Dict]:
        with open(filepath, "r") as f:
            line = f.readline().strip()
            while line != "":
                if re.search(r"\d+\|t\|", line) is not None:
                    pmid, title = line.split("|t|")
                    # The next line has to be the abstract.
                    abstract = f.readline().split("|a|")[-1].strip()
                    line = f.readline().strip()
                    mentions = []
                    while line != "":
                        split_line = line.split("\t")
                        if len(split_line) == 6:
                            _, start, end, text, type_, concept_id = split_line
                        # This entity is not grounded.
                        elif len(split_line) == 5:
                            _, start, end, text, type_, concept_id = *split_line, None
                        # This entity is not grounded and has no type.
                        else:
                            _, start, end, text, type_, concept_id = *split_line, None, None

                        mentions.append(
                            {"concept_id": concept_id, "type": type_, "text": text, "offsets": [int(start), int(end)]}
                        )
                        line = f.readline().strip()
                    yield {"pmid": pmid, "title": title, "abstract": abstract, "mentions": mentions}
                    line = f.readline().strip()

    def _pubtator_parse_to_bigbio_kb(self, pubtator_parse: Dict) -> Dict:
        """
        Transform a PubTator parse (conforming to the standard PubTator schema) obtained with
        `_parse_pubtator_file_file` into a dictionary conforming to the `bigbio-kb` schema
        (as defined in ../schemas/kb.py)

        :param pubtator_parse:
        """

        unified_example = {}

        unified_example["id"] = pubtator_parse["pmid"]
        unified_example["document_id"] = pubtator_parse["pmid"]

        unified_example["passages"] = [
            {
                "id": pubtator_parse["pmid"] + "_title",
                "type": "title",
                "text": pubtator_parse["title"],
                "offsets": [[0, len(pubtator_parse["title"])]],
            },
            {
                "id": pubtator_parse["pmid"] + "_abstract",
                "type": "abstract",
                "text": pubtator_parse["abstract"],
                "offsets": [
                    [
                        # +1 assumes the title and abstract will be joined by a space.
                        len(pubtator_parse["title"]) + 1,
                        len(pubtator_parse["title"]) + 1 + len(pubtator_parse["abstract"]),
                    ]
                ],
            },
        ]

        unified_entities = {}
        for entity in pubtator_parse["mentions"]:
            # We use the entity type to convert to the unified db_name
            db_name = self.PUBTATOR_TYPE_TO_UNIFIED_DB_NAME[entity["type"]]
            # PubTator prefixes MeSH IDs with "MESH:", so we strip it here.
            db_id = entity["concept_id"].replace("MESH:", "")
            if entity["concept_id"] not in unified_entities:
                unified_entities[entity["concept_id"]] = {
                    "id": entity["concept_id"],
                    "type": entity["type"],
                    "text": [entity["text"]],
                    "offsets": [entity["offsets"]],
                    "normalized": [{"db_name": db_name, "db_id": db_id}],
                }
            else:
                unified_entities[entity["concept_id"]]["text"].append(entity["text"])
                unified_entities[entity["concept_id"]]["offsets"].append(entity["offsets"])
                unified_entities[entity["concept_id"]]["normalized"].append({"db_name": db_name, "db_id": db_id})

        return unified_example
