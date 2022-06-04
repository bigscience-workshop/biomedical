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
This dataset contains the CRAFT corpus, a collection of 97 articles from the PubMed Central Open Access subset, 
each of which has been annotated along a number of different axes spanning structural, coreference, and concept annotation. 
Due to current limitations of the current schema, corefs are not included in this dataloader. They will be implemented in a future version
"""

import os
from pydoc import doc
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET
import datasets
from numpy import source
from yaml import parse
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks, Lang
import zipfile

_LOCAL = False
_LANGUAGES = [Lang.EN]
_CITATION = """\
@article{bada2012concept,
  title={Concept annotation in the CRAFT corpus},
  author={Bada, Michael and Eckert, Miriam and Evans, Donald and Garcia, Kristin and Shipley, Krista and Sitnikov, Dmitry and Baumgartner, William A and Cohen, K Bretonnel and Verspoor, Karin and Blake, Judith A and others},
  journal={BMC bioinformatics},
  volume={13},
  number={1},
  pages={1--20},
  year={2012},
  publisher={BioMed Central}
}
"""

_DATASETNAME = "craft"


_DESCRIPTION = """
This dataset contains the CRAFT corpus, a collection of 97 articles from the PubMed Central Open Access subset, 
each of which has been annotated along a number of different axes spanning structural, coreference, and concept annotation.
Due to current limitations of the current schema, corefs are not included in this dataloader. They will be implemented in a future version
"""

_HOMEPAGE = "https://pubmed.ncbi.nlm.nih.gov/22776079/"

_LICENSE = "CC3.0"

_URL = {
    _DATASETNAME: "https://github.com/UCDenver-ccp/CRAFT/archive/refs/tags/v5.0.0.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "5.0.0"

_BIGBIO_VERSION = "1.0.0"

_CLASS_LABELS = {
    "CHEBI": "Chemical Entities of Biological Interest ",
    "CL": "Cell Ontology",
    "GO_BP": "Gene Ontology Biological Process",
    "GO_CC": "Gene Ontology Cellular Component",
    "GO_MF": "Gene Ontology Molecular Function",
    "MONDO": "MONDO Disease Ontology",
    "MOP": "Molecular Process Ontology",
    "NCBITaxon": "NCBI Taxonomy",
    "PR": "Protein Ontology",
    "SO": "Sequence Ontology ",
    "UBERON": "Uberon ",
}


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
#  Append "Dataset" to the class name: BioASQ --> BioasqDataset
class CraftDataset(datasets.GeneratorBasedBuilder):
    """This dataset presents the concept annotations of the Colorado Richly Annotated Full-Text (CRAFT) Corpus, a collection of 97 full-length,
    open-access biomedical journal articles that have been annotated both semantically and syntactically to serve as a research resource for the
     biomedical natural-language-processing (NLP) community. CRAFT identifies all mentions of nearly all concepts from nine prominent biomedical
     ontologies and terminologies: the Cell Type Ontology, the Chemical Entities of Biological Interest ontology, the NCBI Taxonomy, the Protein
      Ontology, the Sequence Ontology, the entries of the Entrez Gene database, and the three subontologies of the Gene Ontology."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    bigbio_schema_name = "kb"
    BUILDER_CONFIGS = [
        BigBioConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        BigBioConfig(
            name=f"{_DATASETNAME}_bigbio_{bigbio_schema_name}",
            version=BIGBIO_VERSION,
            description=f"{_DATASETNAME} BigBio schema",
            schema=f"bigbio_{bigbio_schema_name}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]
    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`
        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "offsets": datasets.Sequence(datasets.Value("int64")),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_[bigbio_schema_name]":
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
        text_subdir = r"CRAFT-5.0.0\articles\txt"
        urls = _URL[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "text_dir": os.path.join(data_dir, text_subdir),
                },
            ),
        ]

    def _read_text(self, file) -> str:
        """
        Read text from the article and return it
        """
        with open(file, "r") as f:
            text = f.read()
        return text

    def _read_ann(self, file, ann_type) -> dict:
        tree = ET.parse(file)
        root = tree.getroot()
        entities = []
        for ann in root.findall("annotation"):
            id = ann.find("mention").attrib["id"]
            span = ann.find("span")
            start, end = span.attrib["start"], span.attrib["end"]
            text = ann.find("spannedText").text
            entity = {
                "entity_id": id,
                "offsets": [start, end],
                "type": ann_type,
                "text": text,
            }
            entities.append(entity)
        return entities

    def entity_to_bigbio_schema(self, entity):
        bigbio_entity = {
            "id": entity["doc_id"],
            "offsets": [entity["offsets"]],
            "text": [entity["text"]],
            "type": entity["type"],
            "normalized": [],
        }
        return bigbio_entity

    def _generate_examples(self, data_dir, text_dir) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        ner_dirs = {
            key: os.path.join(
                r"CRAFT-5.0.0\concept-annotation",
                key,
                key + "+extensions",
                "knowtator",
            )
            for key in _CLASS_LABELS.keys()
        }
        text_file_list = [
            file for file in os.listdir(text_dir) if file.split(".")[-1] == "txt"
        ]
        for filename in text_file_list:
            doc_id = filename.split(".")[0]
            entities = []
            article_text = self._read_text(os.path.join(text_dir, filename))
            print(doc_id, article_text[:20], end="\n")
            for ann_type, ann_dir in ner_dirs.items():
                ann_file = os.path.join(data_dir, ann_dir, filename + ".knowtator.xml")
                print(ann_file)
                entities.extend(self._read_ann(ann_file, ann_type))
            if self.config.schema == "source":
                source_example = {
                    "doc_id": doc_id,
                    "text": article_text,
                    "entities": entities,
                }
                yield doc_id, source_example

            elif self.config.schema == "bigbio_[bigbio_schema_name]":
                bigbio_example = {
                    "id": doc_id,
                    "document_id": doc_id,
                    "passages": [
                        {
                            "id": doc_id + "_text",
                            "type": "text",
                            "text": [article_text],
                            "offsets": [0, len(article_text)],
                        }
                    ],
                    "entities": [
                        self.entity_to_bigbio_schema(entity) for entity in entities
                    ],
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }
                yield doc_id, bigbio_example


if __name__ == "__main__":
    datasets.load_dataset(__file__)
