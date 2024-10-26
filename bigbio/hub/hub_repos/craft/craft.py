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
from typing import Dict, Iterator, List
from xml.etree import ElementTree as ET

import datasets

from .bigbiohub import BigBioConfig, Tasks, kb_features

_LOCAL = False
_LANGUAGES = ["English"]
_PUBMED = True
_CITATION = """\
@article{bada2012concept,
  title={Concept annotation in the CRAFT corpus},
  author={Bada, Michael and Eckert, Miriam and Evans, Donald and Garcia, Kristin and Shipley, Krista and Sitnikov, \
    Dmitry and Baumgartner, William A and Cohen, K Bretonnel and Verspoor, Karin and Blake, Judith A and others},
  journal={BMC bioinformatics},
  volume={13},
  number={1},
  pages={1--20},
  year={2012},
  publisher={BioMed Central}
}
"""

_DATASETNAME = "craft"
_DISPLAYNAME = "CRAFT"

_DESCRIPTION = """
This dataset contains the CRAFT corpus, a collection of 97 articles from the PubMed Central Open Access subset,
each of which has been annotated along a number of different axes spanning structural, coreference, and concept
annotation. Due to current limitations of the current schema, corefs are not included in this dataloader.
They will be implemented in a future version.
"""

_HOMEPAGE = "https://github.com/UCDenver-ccp/CRAFT"

_LICENSE = "CC_BY_3p0_US"

_URL = {
    "source": "https://github.com/UCDenver-ccp/CRAFT/archive/refs/tags/v5.0.2.zip",
    "bigbio_kb": "https://github.com/UCDenver-ccp/CRAFT/archive/refs/tags/v5.0.2.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "5.0.2"
_BIGBIO_VERSION = "1.0.0"

_CONCEPT_ANNOTATIONS = {
    "CHEBI": "Chemical Entities of Biological Interest ",
    "CL": "Cell Ontology",
    "GO_BP": "Gene Ontology Biological Process",
    "GO_CC": "Gene Ontology Cellular Component",
    "GO_MF": "Gene Ontology Molecular Function",
    "MONDO": "MONDO Disease Ontology",
    "MOP": "Molecular Process Ontology",
    "NCBITaxon": "NCBI Taxonomy",
    "PR": "Protein Ontology",
    "SO": "Sequence Ontology",
    "UBERON": "Uberon",
}

logger = datasets.utils.logging.get_logger(__name__)


class CraftDataset(datasets.GeneratorBasedBuilder):
    """
    This dataset presents the concept annotations of the Colorado Richly Annotated Full-Text (CRAFT) Corpus, a
    collection of 97 full-length, open-access biomedical journal articles that have been annotated both semantically
    and syntactically to serve as a research resource for the biomedical natural-language-processing (NLP) community.
    CRAFT identifies all mentions of nearly all concepts from nine prominent biomedical ontologies and terminologies:
        - the Cell Type Ontology,
        - the Chemical Entities of Biological Interest ontology,
        - the NCBI Taxonomy, the Protein Ontology,
        - the Sequence Ontology,
        - the entries of the Entrez Gene database, and t
        - he three subontologies of the Gene Ontology.
    """

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
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "annotations": [
                        {
                            "offsets": datasets.Sequence([datasets.Value("int64")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "db_name": datasets.Value("string"),
                            "db_id": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = kb_features
        else:
            raise NotImplementedError(f"Schema {self.config.schema} not supported")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URL[self.config.schema]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir, "split": "validation"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": data_dir, "split": "test"},
            ),
        ]

    def get_splits(self, data_dir: str) -> Dict:
        """Load `dict[split, list[pmid]]`"""

        splits_dir = os.path.join(data_dir, f"CRAFT-{_SOURCE_VERSION}", "articles", "ids")
        splits = {}
        for split in ["train", "dev", "test"]:
            with open(os.path.join(splits_dir, f"craft-ids-{split}.txt")) as fp:
                split_name = "validation" if split == "dev" else split
                splits[split_name] = [line.strip() for line in fp.readlines()]
        return splits

    def get_texts(self, data_dir: str) -> Dict:
        """Load dict[pmid,text]"""

        texts_dir = os.path.join(data_dir, f"CRAFT-{_SOURCE_VERSION}", "articles", "txt")
        documents = {}
        for file in os.listdir(texts_dir):
            if not file.endswith(".txt"):
                continue

            pmid = file.replace(".txt", "")
            with open(os.path.join(texts_dir, file)) as fp:
                documents[pmid] = fp.read()

        return documents

    def _extract_mondo_annotations(self, path: str) -> Iterator[Dict]:
        """Extract MONDO annotations"""
        root = ET.parse(path)
        for a in root.findall("document/annotation"):
            span = a.find("span")
            assert span is not None

            start = span.attrib["start"]
            end = span.attrib["end"]

            ea = {
                "offsets": [[start, end]],
                "text": [span.text],
            }

            normalization = a.find("class")
            if normalization is not None:
                mondo_id = normalization.attrib["id"].replace("http://purl.obolibrary.org/obo/", "")
                mondo_id = mondo_id.replace("_", ":")
                ea["db_id"] = mondo_id

            yield ea

    def _extract_other_annotations(self, path: str) -> Iterator[Dict]:
        """Extract all other annotations (CHEBI, UBERON, ...)"""

        # NOTE: handle knowtator normalization format
        # <annotation>
        #     <mention id="UBERON_Instance_30000" />
        # </annotation>
        # <classMention id="UBERON_Instance_30166">
        #     <mentionClass id="UBERON:0002435">striatum</mentionClass>
        # </classMention>

        root = ET.parse(path)
        instance_to_db_id = {
            e.attrib["id"]: e.find("mentionClass").attrib["id"]
            for e in root.findall("classMention")
            if e.find("mentionClass") is not None
        }

        for a in root.findall("annotation"):
            span = a.find("span")
            assert span is not None
            offsets = [[span.attrib["start"], span.attrib["end"]] for span in a.findall("span")]
            text = a.find("spannedText").text.split(" ... ")
            ea = {"offsets": offsets, "text": text}
            mention = a.find("mention")
            db_id = None
            if mention is not None:
                instance = mention.attrib["id"]
                db_id = instance_to_db_id.get(instance)
                ea["db_id"] = db_id

            yield ea

    def get_annotations(self, data_dir: str) -> Dict:
        """Load dict[pmid,annotations]"""

        annotations_dir = os.path.join(data_dir, f"CRAFT-{_SOURCE_VERSION}", "concept-annotation")

        annotations: Dict = {}
        for concept in _CONCEPT_ANNOTATIONS:
            if concept == "MONDO":
                folder = os.path.join(
                    annotations_dir,
                    "MONDO",
                    "MONDO_without_genotype_annotations",
                    "knowtator-2",
                )
            else:
                folder = os.path.join(
                    annotations_dir,
                    concept,
                    concept,
                    "knowtator",
                )

            for file in sorted(os.listdir(folder)):
                pmid = file.replace(".xml", "").replace(".txt", "").replace(".knowtator", "")
                path = os.path.join(folder, file)

                if pmid not in annotations:
                    annotations[pmid] = []

                annotations_generator = (
                    self._extract_mondo_annotations(path)
                    if concept == "MONDO"
                    else self._extract_other_annotations(path)
                )

                for a in annotations_generator:
                    a["db_name"] = concept
                    annotations[pmid].append(a)

        return annotations

    def _generate_examples(self, data_dir: str, split: str):
        """Yields examples as (key, example) tuples."""

        splits = self.get_splits(data_dir=data_dir)
        texts = self.get_texts(data_dir=data_dir)
        annotations = self.get_annotations(data_dir=data_dir)

        if self.config.schema == "source":
            for pmid in splits[split]:
                example = {
                    "pmid": pmid,
                    "text": texts[pmid],
                    "annotations": annotations[pmid],
                }
                yield pmid, example

        elif self.config.schema == "bigbio_kb":
            uid = map(str, itertools.count(start=0, step=1))
            for pmid in splits[split]:
                example = {
                    "id": next(uid),
                    "document_id": pmid,
                    "passages": [
                        {
                            "id": next(uid),
                            "type": "text",
                            "text": [texts[pmid]],
                            "offsets": [[0, len(texts[pmid])]],
                        }
                    ],
                    "entities": [
                        {
                            "id": next(uid),
                            "offsets": a["offsets"],
                            "text": a["text"],
                            "type": a["db_name"],
                            "normalized": [{"db_name": a["db_name"], "db_id": a["db_id"]}],
                        }
                        for a in annotations[pmid]
                    ],
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }
                yield next(uid), example
