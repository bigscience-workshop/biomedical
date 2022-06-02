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
The extended Anatomical Entity Mention corpus (AnatEM) consists of 1212 documents
(approx. 250,000 words) manually annotated to identify over 13,000 mentions of anatomical
entities. Each annotation is assigned one of 12 granularity-based types such as Cellular
component, Tissue and Organ, defined with reference to the Common Anatomy Reference Ontology
(see https://bioportal.bioontology.org/ontologies/CARO).
"""
from pathlib import Path
from typing import Dict, Iterator, Tuple

import datasets

import bigbio.utils.parsing as parsing
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{pyysalo2014anatomical,
  title={Anatomical entity mention recognition at literature scale},
  author={Pyysalo, Sampo and Ananiadou, Sophia},
  journal={Bioinformatics},
  volume={30},
  number={6},
  pages={868--875},
  year={2014},
  publisher={Oxford University Press}
}
"""

_DATASETNAME = "anat_em"

_DESCRIPTION = """\
The extended Anatomical Entity Mention corpus (AnatEM) consists of 1212 documents (approx. 250,000 words)
manually annotated to identify over 13,000 mentions of anatomical entities. Each annotation is assigned one
of 12 granularity-based types such as Cellular component, Tissue and Organ, defined with reference to the
Common Anatomy Reference Ontology.
"""

_HOMEPAGE = "http://nactem.ac.uk/anatomytagger/#AnatEM"
_LICENSE = "Creative Commons BY-SA 3.0 license"

_URLS = {_DATASETNAME: "http://nactem.ac.uk/anatomytagger/AnatEM-1.0.2.tar.gz"}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.2"
_BIGBIO_VERSION = "1.0.0"


class AnatEMDataset(datasets.GeneratorBasedBuilder):
    """The extended Anatomical Entity Mention corpus (AnatEM)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="anat_em_source",
            version=SOURCE_VERSION,
            description="AnatEM source schema",
            schema="source",
            subset_id="anat_em",
        ),
        BigBioConfig(
            name="anat_em_bigbio_kb",
            version=BIGBIO_VERSION,
            description="AnatEM BigBio schema",
            schema="bigbio_kb",
            subset_id="anat_em",
        ),
    ]

    DEFAULT_CONFIG_NAME = "anat_em_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "document_type": datasets.Value("string"),  # Either PMC or PM
                    "text": datasets.Value("string"),
                    "text_type": datasets.Value(
                        "string"
                    ),  # Either abstract (for PM) or sec / caption (for PMC)
                    "entities": [
                        {
                            "entity_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
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

    def _split_generators(self, dl_manager):
        urls = _URLS[_DATASETNAME]
        data_dir = Path(dl_manager.download_and_extract(urls))

        standoff_dir = data_dir / "AnatEM-1.0.2" / "standoff"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split_dir": standoff_dir / "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split_dir": standoff_dir / "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split_dir": standoff_dir / "devel"},
            ),
        ]

    def _generate_examples(self, split_dir: Path) -> Iterator[Tuple[str, Dict]]:
        if self.config.name == "anat_em_source":
            for file in split_dir.iterdir():
                # Ignore hidden files and annotation files - we just consider the brat text files
                if file.name.startswith("._") or file.name.endswith(".ann"):
                    continue

                # Read brat annotations for the given text file and convert example to the source format
                brat_example = parsing.parse_brat_file(file)
                source_example = self._to_source_example(file, brat_example)

                yield source_example["document_id"], source_example

        elif self.config.name == "anat_em_bigbio_kb":
            for file in split_dir.iterdir():
                # Ignore hidden files and annotation files - we just consider the brat text files
                if file.name.startswith("._") or file.name.endswith(".ann"):
                    continue

                # Read brat annotations for the given text file and convert example to the BigBio-KB format
                brat_example = parsing.parse_brat_file(file)
                kb_example = parsing.brat_parse_to_bigbio_kb(brat_example)
                kb_example["id"] = kb_example["document_id"]

                # Fix text type annotation for the converted example
                _, text_type = self.get_document_type_and_text_type(file)
                kb_example["passages"][0]["type"] = text_type

                yield kb_example["id"], kb_example

    def _to_source_example(self, input_file: Path, brat_example: Dict) -> Dict:
        """
        Converts an example extracted using the default brat parsing logic to the source format
        of the given corpus.
        """
        document_type, text_type = self.get_document_type_and_text_type(input_file)

        source_example = {
            "document_id": brat_example["document_id"],
            "document_type": document_type,
            "text": brat_example["text"],
            "text_type": text_type,
        }

        id_prefix = brat_example["document_id"] + "_"

        source_example["entities"] = []
        for entity_annotation in brat_example["text_bound_annotations"]:
            entity_ann = entity_annotation.copy()

            entity_ann["entity_id"] = id_prefix + entity_ann["id"]
            entity_ann.pop("id")

            source_example["entities"].append(entity_ann)

        return source_example

    def get_document_type_and_text_type(self, input_file: Path) -> Tuple[str, str]:
        """
        Extracts the document type (PubMed(PM) or PubMedCentral (PMC)) and the respective
        text type (abstract for PM and sec or caption for (PMC) from the name of the given
        file, e.g.:

        PMID-9778569.txt -> ("PM", "abstract")

        PMC-1274342-sec-02.txt -> ("PMC", "sec")

        PMC-1592597-caption-02.ann -> ("PMC", "caption")

        """
        name_parts = str(input_file.stem).split("-")

        if name_parts[0] == "PMID":
            return "PM", "abstract"

        elif name_parts[0] == "PMC":
            return "PMC", name_parts[2]
        else:
            raise AssertionError(f"Unexpected file prefix {name_parts[0]}")
