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

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

# TODO: Add BibTeX citation
_LOCAL = False
_CITATION = """\
@article{,
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

_DESCRIPTION = """\
SNP normalization corpus downloaded from (http://www.scai.fraunhofer.de/snp-normalization-corpus.html).
SNPs are associated with unambiguous dbSNP identifiers.
"""

_HOMEPAGE = "http://www.scai.fraunhofer.de/snp-normalization-corpus.html"

_LICENSE = """
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

# 

# this is a backup url in case the official one will stop working
# _URLS = ["http://github.com/rockt/SETH/zipball/master/"]
_URLS = {
    "source" : "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/normalization-variation-corpus.gz",
    "bigbio_kb" : "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/normalization-variation-corpus.gz"
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

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    # TODO: For each dataset, implement Config for Source and BigBio;
    #  If dataset contains more than one subset (see examples/bioasq.py) implement for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. bioasq10b_(source|bigbio)_[bigbio_schema_name]
    #   - version: option = (SOURCE_VERSION|BIGBIO_VERSION)
    #   - description: one line description for the dataset
    #   - schema: options = (source|bigbio_[bigbio_schema_name])
    #   - subset_id: subset id is the canonical name for the dataset (eg. bioasq10b)
    #  where [bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="thomas2011_source",
            version=SOURCE_VERSION,
            description="Thomas et al 2011 source schema",
            schema="source",
            subset_id="thomas2011",
        ),
        BigBioConfig(
            name="thomas2011_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Thomas et al 2011 BigBio schema",
            schema="bigbio_kb",
            subset_id="thomas2011",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thomas2011_source"

    _ENTITY_TYPES = {"Nucleotide Sequence Mutation", "Protein Sequence Mutation"}

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.
        # Much of this design is copied from biodatasets/verspoor_2013/verspoor_2013.py

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

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
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        # Download gets entire git repo containing unused data from other datasets
        # repo_dir = Path(dl_manager.download_and_extract(_URLS[0]))
        # data_dir = repo_dir / "data"
        # data_dir.mkdir(exist_ok=True)

        # Find the relevant files from Verspor2013 and move them to a new directory
        # thomas2011_files = repo_dir.glob("*/*/*thomas2011/**/*")
        # for file in thomas2011_files:
        #    if file.is_file() and "README" not in str(file):
        #        file.rename(data_dir / file.name)

        # Delete all unused files and directories from the original download
        #for x in repo_dir.glob("[!data]*"):
        #    if x.is_file():
        #        x.unlink()
        #    elif x.is_dir():
        #        rmtree(x)

        data_dir = dl_manager.download_and_extract(_URLS[self.config.schema])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'annotations.txt'),
                    "split": "test",
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.
    def _generate_examples(self, filepath: str, split: str) -> Tuple[int, Dict]:

        """Yields examples as (key, example) tuples."""
        if split == "test":
            data_ann = []
            with open(filepath, encoding="utf-8") as ann_tsv_file:
                csv_reader_code = csv.reader(
                    ann_tsv_file, quotechar="'", delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True
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
            for id_ in df.doc_id.unique():
                elist = []
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
                            "offsets": [[int(row.off1), int(row.off2)]],
                            "normalized": [{"db_name": "dbSNP", "db_id": row.dbSNP_id}],
                        }
                    )
                yield id_, {
                    "id": id_,  # uid is an unique identifier for every record that starts from 1
                    "document_id": str(row[0]),
                    "entities": elist,
                    "passages": [],
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
