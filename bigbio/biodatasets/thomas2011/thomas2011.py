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
SNP Corpus Version 1.0 (as of March 17, 2011) - Copied from http://www.scai.fraunhofer.de/snp-normalization-corpus.html

The corpus consists of 296 Medline citations. Citations were screened for mutations using
a modified version of MutationFinder. The used regular expressions are available in
'mutationfinder.txt'.

The SNPs (also missed by MutationFinder) were manually annotated with the corresponding
dbSNP identifier, if available. Mutations without a valid dbSNP identifier were omitted.

The corpus consists of 527 mutation rs-pairs. Due to licence restrictions of
MEDLINE, abstracts are not contained in the corpus, but can be downloaded from
MEDLINE using eUtils.

To allow for a reproduction of our corpus, we also provide the original
SNP mention in the abstract.

The corpus can be used to assess the performance of algorithms capable of
associating variation mentions with dbSNP identifiers. It is published for
academic use only and usage for development of commercial products is not
permitted.
}

"""

import csv
import os
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import datasets
import pandas as pd
import requests

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import CustomLicense
import time

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{thomas2011challenges,
  author    = {Thomas, Philippe and Klinger, Roman and Furlong, Laura and Hofmann-Apitius, Martin and Friedrich, Christoph},
  title     = {Challenges in the association of human single nucleotide polymorphism mentions with unique database identifiers},
  journal   = {BMC Bioinformatics},
  volume    = {12},
  year      = {2011},
  url       = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-S4-S4},
  doi       = {https://doi.org/10.1186/1471-2105-12-S4-S4}
}
"""

_DATASETNAME = "thomas2011"
_DISPLAYNAME = "SNP Corpus"

_DESCRIPTION = """\
SNP normalization corpus downloaded from (http://www.scai.fraunhofer.de/snp-normalization-corpus.html).
SNPs are associated with unambiguous dbSNP identifiers.
"""

_HOMEPAGE = "http://www.scai.fraunhofer.de/snp-normalization-corpus.html"

_LICENSE = CustomLicense(
    text="""
LICENSE
1. Copyright of abstracts - Due to license restriction of PubMed(R) this corpus contains only annotations.
To facilitate a reproduction of the original corpus, we include the exact position in the text,
as well as the matching string. The articles are composed of <Title><Whitespace><Whitespace><Abstract>
For detailed description of the corpus and its annotations see README.txt.

2. Copyright of regular expression
License of the original regular expressions is subject to the license agreement at http://mutationfinder.sourceforge.net/license.txt
Also the additional rules are subject to these agreement.

3. Copyright of annotations
The annotations are published for academic use only and usage for development of commercial products is not permitted.
"""
)

# this is a backup url in case the official one will stop working
# _URLS = ["http://github.com/rockt/SETH/zipball/master/"]
_URLS = {
    _DATASETNAME: "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/normalization-variation-corpus.gz",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class Thomas2011Dataset(datasets.GeneratorBasedBuilder):
    """Corpus consists of 296 Medline citations."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="Thomas et al 2011 source schema",
            schema="source",
            subset_id="thomas2011",
        ),
        BigBioConfig(
            name=f"{_DATASETNAME}_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Thomas et al 2011 BigBio schema",
            schema="bigbio_kb",
            subset_id="thomas2011",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "covered_text": datasets.Value("string"),
                    "resolved_name": datasets.Value("string"),
                    "offsets": datasets.Sequence([datasets.Value("int32")]),
                    "dbSNP_id": datasets.Value("string"),
                    "protein_or_nucleotide_sequence_mutation": datasets.Value("string"),
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

        data_dir = dl_manager.download_and_extract(_URLS[_DATASETNAME])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "annotations.txt"),
                },
            )
        ]

    def get_clean_pubmed_abstract(self, id):
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        params = {
            "db": "pubmed",
            "id": id,
            "retmode": "xml",
            "rettype": "medline",
        }
        res = requests.get(url, params=params)
        blank_line_count = 0
        required_text_lines = []
        tree = ET.XML(res.text)
        article = tree.find("PubmedArticle").find("MedlineCitation").find("Article")
        article_title = article.find("ArticleTitle").text
        abstract_parts = [f"{article_title}"]
        article_abstract = article.find("Abstract").findall("AbstractText")
        for abstract_part in article_abstract:
            label = abstract_part.attrib.get("Label", "")
            if label:
                abstract_parts.append(f"{label}: {abstract_part.text}")
            else:
                abstract_parts.append(abstract_part.text)
        return " ".join(abstract_parts)

    def _generate_examples(self, filepath: str) -> Tuple[int, Dict]:

        """Yields examples as (key, example) tuples."""
        data_ann = []
        with open(filepath, encoding="utf-8") as ann_tsv_file:
            csv_reader_code = csv.reader(
                ann_tsv_file,
                quotechar="'",
                delimiter="\t",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )
            for id_, row in enumerate(csv_reader_code):
                data_ann.append(row)

        if self.config.schema == "source":
            for id_, row in enumerate(data_ann):
                yield id_, {
                    "doc_id": row[0],
                    "covered_text": row[1],
                    "resolved_name": row[2],
                    "offsets": [(int(row[3]), int(row[4]))],
                    "dbSNP_id": row[5],
                    "protein_or_nucleotide_sequence_mutation": row[6],
                }
        elif self.config.schema == "bigbio_kb":
            cols = [
                "doc_id",
                "covered_text",
                "resolved_name",
                "off1",
                "off2",
                "dbSNP_id",
                "protein_or_nucleotide_sequence_mutation",
            ]
            df = pd.DataFrame(data_ann, columns=cols)
            uid = 0
            curr_count = 0
            for id_ in df.doc_id.unique():
                curr_count += 1
                if curr_count == 3:
                    # The PubMed API limits 3 requests per second without an API key
                    time.sleep(0.5)
                    curr_count = 0
                elist = []
                abstract_text = self.get_clean_pubmed_abstract(id_)
                uid += 1
                passage = {
                    "id": uid,
                    "type": "",
                    "text": [abstract_text],
                    "offsets": [[0, len(abstract_text)]],
                }
                for row in df.loc[df.doc_id == id_].itertuples():
                    uid += 1
                    if row.protein_or_nucleotide_sequence_mutation == "PSM":
                        ent_type = "Protein Sequence Mutation"
                    else:
                        ent_type = "Nucleotide Sequence Mutation"
                    elist.append(
                        {
                            "id": str(uid),
                            "type": ent_type,
                            "text": [row.covered_text],
                            "offsets": [[int(row.off1) - 1, int(row.off2) - 1]],
                            "normalized": [{"db_name": "dbSNP", "db_id": row.dbSNP_id}],
                        }
                    )
                yield id_, {
                    "id": id_,  # uid is an unique identifier for every record that starts from 1
                    "document_id": str(row[0]),
                    "entities": elist,
                    "passages": [passage],
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }
