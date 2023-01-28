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


import os
from typing import Dict, Iterator, List, Tuple

import bioc
import datasets
import pandas as pd

from .bigbiohub import BigBioConfig, Tasks, kb_features

_LOCAL = False
_PUBMED = True
_LANGUAGES = ["English"]

_CITATION = """\
@inproceedings{arighi2017bio,
  title={Bio-ID track overview},
  author={Arighi, Cecilia and Hirschman, Lynette and Lemberger, Thomas and Bayer, Samuel and Liechti, Robin and Comeau, Donald and Wu, Cathy},
  booktitle={Proc. BioCreative Workshop},
  volume={482},
  pages={376},
  year={2017}
}
"""

_DATASETNAME = "bioid"
_DISPLAYNAME = "BIOID"

_DESCRIPTION = """\
The Bio-ID track focuses on entity tagging and ID assignment to selected bioentity types.
The task is to annotate text from figure legends with the entity types and IDs for taxon (organism), gene, protein, miRNA, small molecules,
cellular components, cell types and cell lines, tissues and organs. The track draws on SourceData annotated figure
legends (by panel), in BioC format, and the corresponding full text articles (also BioC format) provided for context.
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-1/"

_LICENSE = "UNKNOWN"

_URLS = {
    _DATASETNAME: "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
]

_SOURCE_VERSION = "2.0.0"

_BIGBIO_VERSION = "1.0.0"


class BioidDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bioid_source",
            version=SOURCE_VERSION,
            description="bioid source schema",
            schema="source",
            subset_id="bioid",
        ),
        BigBioConfig(
            name="bioid_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bioid BigBio schema",
            schema="bigbio_kb",
            subset_id="bioid",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bioid_source"

    ENTITY_TYPES_NOT_NORMALIZED = [
        "cell",
        "gene",
        "molecule",
        "protein",
        "subcellular",
        "tissue",
        "organism",
    ]

    DB_NAME_TO_ENTITY_TYPE = {
        "BAO": "assay",  # https://www.ebi.ac.uk/ols/ontologies/bao
        "CHEBI": "chemical",
        "CL": "cell",  # https://www.ebi.ac.uk/ols/ontologies/cl
        "Corum": "protein",  # https://mips.helmholtz-muenchen.de/corum/
        "GO": "gene",  # https://geneontology.org/
        "PubChem": "chemical",
        "Rfam": "rna",  # https://rfam.org/
        "Uberon": "anatomy",
        "Cellosaurus": "cell",
        "NCBI gene": "gene",
        "NCBI taxon": "species",
        "Uniprot": "protein",
    }

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.
        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sourcedata_document": datasets.Value("string"),
                    "doi": datasets.Value("string"),
                    "pmc_id": datasets.Value("string"),
                    "figure": datasets.Value("string"),
                    "sourcedata_figure_dir": datasets.Value("string"),
                    "passages": [
                        {
                            "text": datasets.Value("string"),
                            "offset": datasets.Value("int32"),
                            "annotations": [
                                {
                                    "thomas_article": datasets.Value("string"),
                                    "doi": datasets.Value("string"),
                                    "don_article": datasets.Value("int32"),
                                    "figure": datasets.Value("string"),
                                    "annot id": datasets.Value("int32"),
                                    "paper id": datasets.Value("int32"),
                                    "first left": datasets.Value("int32"),
                                    "last right": datasets.Value("int32"),
                                    "length": datasets.Value("int32"),
                                    "byte length": datasets.Value("int32"),
                                    "left alphanum": datasets.Value("string"),
                                    "text": datasets.Value("string"),
                                    "right alphanum": datasets.Value("string"),
                                    "obj": datasets.Value("string"),
                                    "overlap": datasets.Value("string"),
                                    "identical span": datasets.Value("string"),
                                    "overlap_label_count": datasets.Value("int32"),
                                }
                            ],
                        }
                    ],
                }
            )

        # Choose the appropriate bigbio schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.
        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.
        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "bigbio_kb":
            features = kb_features

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

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
        ]

    def load_annotations(self, path: str) -> Dict[str, Dict]:
        """
        We load annotations from `annotations.csv`
        becuase the one in the BioC xml files have offsets issues.
        """

        df = pd.read_csv(path, sep=",")

        df.fillna(-1, inplace=True)

        annotations: Dict[str, Dict] = {}

        for record in df.to_dict("records"):

            article_id = str(record["don_article"])

            if article_id not in annotations:
                annotations[article_id] = {}

            figure = record["figure"]

            if figure not in annotations:
                annotations[article_id][figure] = []

            annotations[article_id][figure].append(record)

        return annotations

    def load_data(self, data_dir: str) -> List[Dict]:
        """
        Compose text from BioC files with annotations from `annotations.csv`.
        We load annotations from `annotations.csv`
        becuase the one in the BioC xml files have offsets issues.
        """

        text_dir = os.path.join(data_dir, "BioIDtraining_2", "caption_bioc")
        annotation_file = os.path.join(data_dir, "BioIDtraining_2", "annotations.csv")

        annotations = self.load_annotations(path=annotation_file)

        data = []

        for file_name in os.listdir(text_dir):

            if file_name.startswith(".") or not file_name.endswith(".xml"):
                continue

            collection = bioc.load(os.path.join(text_dir, file_name))

            for document in collection.documents:

                item = document.infons

                assert (
                    len(document.passages) == 1
                ), "Document contains more than one passage (figure caption). This is not expected!"

                passage = document.passages[0]

                article_id = document.infons["pmc_id"]
                figure = document.infons["sourcedata_figure_dir"]

                try:
                    passage.annotations = annotations[article_id][figure]
                except KeyError:
                    passage.annotations = []

                item["passages"] = [
                    {
                        "text": passage.text,
                        "annotations": passage.annotations,
                        "offset": passage.offset,
                    }
                ]

                data.append(item)

        return data

    def get_entity(self, normalization: str) -> Tuple[str, List[Dict]]:
        """
        Compile normalization information from annotation
        """

        db_name_ids = normalization.split(":")

        db_ids = None

        # ids from cellosaurus do not have db name
        if len(db_name_ids) == 1:
            db_name = "Cellosaurus"
            db_ids = db_name_ids[0].split("|")
        else:
            # quirk
            if db_name_ids[0] == "CVCL_6412|CL":
                db_name = "Cellosaurus"
                db_ids = ["CVCL_6412"]
            else:
                db_name = db_name_ids[0]
                # db_name hints for entity type: skip if does not provide normalization
                if db_name not in self.ENTITY_TYPES_NOT_NORMALIZED:
                    # Uberon:UBERON:0001891
                    # NCBI gene:9341
                    db_id_idx = 2 if db_name == "Uberon" else 1
                    db_ids = [i.split(":")[db_id_idx] for i in normalization.split("|")]

        normalized = (
            [{"db_name": db_name, "db_id": i} for i in db_ids]
            if db_ids is not None
            else []
        )

        # ideally we should have canonical entity types w/ a  dedicated enum like `Tasks`

        if db_name in self.ENTITY_TYPES_NOT_NORMALIZED:
            entity_type = db_name
        else:
            entity_type = self.DB_NAME_TO_ENTITY_TYPE[db_name]

        return entity_type, normalized

    def _generate_examples(
        self, data_dir: str, split: str
    ) -> Iterator[Tuple[int, Dict]]:
        """Yields examples as (key, example) tuples."""

        data = self.load_data(data_dir=data_dir)

        if self.config.schema == "source":
            for uid, document in enumerate(data):
                yield uid, document

        elif self.config.schema == "bigbio_kb":

            uid = 0  # global unique id

            for document in data:

                kb_document = {
                    "id": uid,
                    "document_id": document["pmc_id"],
                    "passages": [],
                    "entities": [],
                    "relations": [],
                    "events": [],
                    "coreferences": [],
                }

                uid += 1

                for passage in document["passages"]:
                    kb_document["passages"].append(
                        {
                            "id": uid,
                            "type": "figure_caption",
                            "text": [passage["text"]],
                            "offsets": [[0, len(passage["text"])]],
                        }
                    )
                    uid += 1

                    for a in passage["annotations"]:

                        entity_type, normalized = self.get_entity(a["obj"])

                        kb_document["entities"].append(
                            {
                                "id": uid,
                                "text": [a["text"]],
                                "type": entity_type,
                                "offsets": [[a["first left"], a["last right"]]],
                                "normalized": normalized,
                            }
                        )

                        uid += 1

                yield uid, kb_document
