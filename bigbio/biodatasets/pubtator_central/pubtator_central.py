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


from typing import Dict, Iterator, List, Tuple

import datasets
from bioc import pubtator

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
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
_LICENSE = """\
PUBLIC DOMAIN NOTICE
National Center for Biotechnology Information

This software/database is a "United States Government Work" under the terms of the United States Copyright Act.
It was written as part of the authors' official duties as a United States Government employee and thus cannot be
copyrighted. This software/database is freely available to the public for use. The National Library of Medicine
and the U.S. Government have not placed any restriction on its use or reproduction.

Although all reasonable efforts have been taken to ensure the accuracy and reliability of the software and data,
the NLM and the U.S. Government do not and cannot warrant the performance or results that may be obtained by
using this software or data. The NLM and the U.S. Government disclaim all warranties, express or implied,
including warranties of performance, merchantability or fitness for any particular purpose.
"""

_URLS = {
    "sample": "https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.sample",
    "full": "https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.gz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "2022.01.08"
_BIGBIO_VERSION = "1.0.0"

# Maps the entity types in PubTator to the name of the database they are grounded to
_TYPE_TO_DB_NAME = {
    "Gene": "ncbi_gene",
    "Disease": "mesh",
    "Species": "ncbi_taxon",
    "Chemical": "mesh",
    "CellLine": "cellosaurus",
}

_DB_NAME_TO_URL = {
    "ncbi_gene": "https://www.ncbi.nlm.nih.gov/gene/",
    "mesh": "https://www.nlm.nih.gov/mesh/meshhome.html",
    "ncbi_taxon": "https://www.ncbi.nlm.nih.gov/taxonomy/",
    "cellosaurus": "https://web.expasy.org/cellosaurus/",
    "ncbi_dbsnp": "https://www.ncbi.nlm.nih.gov/snp/",
    "tmvar": "https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/",
}


class PubtatorCentralDataset(datasets.GeneratorBasedBuilder):
    """PubTator Central"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        # sample source
        BigBioConfig(
            name="pubtator_central_sample_source",
            version=SOURCE_VERSION,
            description="PubTator Central sample source schema",
            schema="source",
            subset_id="pubtator_central_sample",
        ),
        # sample big bio
        BigBioConfig(
            name="pubtator_central_sample_bigbio_kb",
            version=BIGBIO_VERSION,
            description="PubTator Central sample BigBio schema",
            schema="bigbio_kb",
            subset_id="pubtator_central_sample",
        ),
        # full dataset source
        BigBioConfig(
            name="pubtator_central_source",
            version=SOURCE_VERSION,
            description="PubTator Central source schema",
            schema="source",
            subset_id="pubtator_central",
        ),
        # full dataset bigbio
        BigBioConfig(
            name="pubtator_central_bigbio_kb",
            version=BIGBIO_VERSION,
            description="PubTator Central BigBio schema",
            schema="bigbio_kb",
            subset_id="pubtator_central",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pubtator_central_source"

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
        urls = _URLS["sample"] if self.config.subset_id.endswith("sample") else _URLS["full"]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: str, split: str) -> Iterator[Tuple[str, Dict]]:
        if self.config.schema == "source":
            for source_example in self._pubtator_to_source(filepath):
                yield source_example["pmid"], source_example

        elif self.config.schema == "bigbio_kb":
            for kb_example in self._pubtator_to_bigbio_kb(filepath):
                yield kb_example["id"], kb_example

    @staticmethod
    def _pubtator_to_source(filepath: Dict) -> Iterator[Dict]:
        with open(filepath, "r") as f:
            for doc in pubtator.iterparse(f):
                source_example = {
                    "pmid": doc.pmid,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "mentions": [
                        {
                            "concept_id": mention.id,
                            "type": mention.type,
                            "text": mention.text,
                            "offsets": [mention.start, mention.end],
                        }
                        for mention in doc.annotations
                    ],
                }
                yield source_example

    def _pubtator_to_bigbio_kb(self, filepath: Dict) -> Iterator[Dict]:
        with open(filepath, "r") as f:
            unified_example = {}
            for doc in pubtator.iterparse(f):
                unified_example["id"] = doc.pmid
                unified_example["document_id"] = doc.pmid

                unified_example["passages"] = [
                    {
                        "id": doc.pmid + "_title",
                        "type": "title",
                        "text": [doc.title],
                        "offsets": [[0, len(doc.title)]],
                    },
                    {
                        "id": doc.pmid + "_abstract",
                        "type": "abstract",
                        "text": [doc.abstract],
                        "offsets": [
                            [
                                # +1 assumes the title and abstract will be joined by a space.
                                len(doc.title) + 1,
                                len(doc.title) + 1 + len(doc.abstract),
                            ]
                        ],
                    },
                ]

                unified_entities = []
                for i, entity in enumerate(doc.annotations):
                    # We need a unique identifier for this entity, so build it from the document id and entity id
                    unified_entity_id = "_".join([doc.pmid, entity.id, str(i)])
                    # Determining db_name is tricky so use a helper to determine this from the entity annotation
                    db_name = self._get_db_name(entity)
                    unified_entities.append(
                        {
                            "id": unified_entity_id,
                            "type": entity.type,
                            "text": [entity.text],
                            "offsets": [[entity.start, entity.end]],
                            "normalized": [{"db_name": db_name, "db_id": entity.id}],
                        }
                    )

                unified_example["entities"] = unified_entities
                unified_example["relations"] = []
                unified_example["events"] = []
                unified_example["coreferences"] = []

                yield unified_example

    @staticmethod
    def _get_db_name(entity: pubtator.PubTatorAnn) -> str:
        if entity.type in _TYPE_TO_DB_NAME:
            db_name = _TYPE_TO_DB_NAME[entity.type]
        elif entity.type in ["Mutation", "ProteinMutation", "DNAMutation"]:
            # Mutation anntotations are grounded to either tmVar or dbSNP
            if entity.id.startswith("tmVar"):
                db_name = "tmVar"
            else:
                db_name = "ncbi_dbsnp"
        else:
            db_name = "unknown"
        return db_name
