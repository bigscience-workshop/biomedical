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

from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
from lxml import etree

from .bigbiohub import BigBioConfig, Tasks, kb_features

_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = False

_CITATION = """\
@article{,
title = {ChEBI: a database and ontology for chemical entities of biological interest},
author = {Degtyarenko, Kirill and de Matos, Paula and Ennis, Marcus and Hastings, Janna and Zbinden, Martin and \
    McNaught, Alan and Alcántara, Rafael and Darsow, Michael and Guedj, Mickaël and Ashburner, Michael},
doi = {10.1093/nar/gkm791},
number = {Database issue},
volume = {36},
month = {January},
year = {2008},
journal = {Nucleic acids research},
issn = {0305-1048},
pages = {D344—50},
url = {https://europepmc.org/articles/PMC2238832},
biburl = {https://aclanthology.org/W19-5008.bib},
bibsource = {https://aclanthology.org/W19-5008/}
}
"""

_DATASETNAME = "chebi"
_DISPLAYNAME = "Chebi"

_DESCRIPTION = """\
ChEBI Chapti contains the results of a collaboration between the European Patent Office and
the ChEBI team. The goal of the project was to identify chemicals within patents and cross-
reference them to ChEBI. The teams manually annotated chemicals in a set of 40 patents.
This was used to measure the performance of the various text-mining tools. This set of
40 patents is distributed in this directory. The results of this work can be seen on the ChEBI
website.
"""

_HOMEPAGE = "https://sourceforge.net/projects/chebi/"
_LICENSE = "CC_BY_SA_4p0"

DATA_URL = "https://github.com/bigscience-workshop/biomedical/files/8568960/PatentAnnotations_GoldStandard.tar.gz"
_URLS = {
    # The original dataset is hosted on CVS on sourceforge. Hence I have downloaded and reuploded it as tar.gz format.
    # Converted via the following command:
    # cvs -z3 -d:pserver:anonymous@a.cvs.sourceforge.net:/cvsroot/chebi co \
    #   chapati/patentsGoldStandard/PatentAnnotations_GoldStandard.tgz
    # mkdir -p ./MoNERo
    # pushd ./MoNERo && 7z x ../MoNERo_2019.7z && popd
    # tar -czf MoNERo.tar.gz ./MoNERo
    _DATASETNAME: DATA_URL,
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class ChebiDataset(datasets.GeneratorBasedBuilder):
    """ChEBI Chapti: Patents dataset for NER and Entity Linking."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        BigBioConfig(
            name=f"{_DATASETNAME}_bigbio_kb",
            version=BIGBIO_VERSION,
            description=f"{_DATASETNAME} BigBio schema",
            schema="bigbio_kb",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "phrase": datasets.Value("string"),
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "attrs": {
                                "chebi-id": datasets.Value("string"),
                                "comment": datasets.Value("string"),
                                "epochem-id": datasets.Value("string"),
                                "id": datasets.Value("string"),
                                "name": datasets.Value("string"),
                                "relevant": datasets.Value("string"),
                                "type": datasets.Value("string"),
                            },
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = kb_features
        else:
            raise NotImplementedError(f"Schema {self.config.schema} is not supported")

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
        data_dir = Path(data_dir) / "scrapbook"
        file_paths = list(sorted(data_dir.glob("./*/source.xml")))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file_paths": file_paths,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, file_paths, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            for filepath in file_paths:
                key, example = self._read_example_from_file(filepath)
                yield key, example

        elif self.config.schema == "bigbio_kb":
            for filepath in file_paths:
                key, example = self._read_example_from_file_in_kb_schema(filepath)
                yield key, example

    def _parse_paragraph(self, para, start=0):
        para_text = []
        entities = []
        for e in para.iter():
            para_text.append(e.text)
            if e.tag == "ne":
                entity = {
                    "phrase": e.text,
                    "start": start,
                    "end": start + len(e.text),
                    "attrs": dict(e.attrib),
                }
                entities.append(entity)
            start += len(e.text)
            if e.tail:
                para_text.append(e.tail)
                start += len(e.tail)
        return "".join(para_text), entities

    def _read_example_from_file(self, filepath: Path) -> Tuple[str, Dict]:
        with open(filepath, encoding="utf-8") as fp:
            xml = etree.fromstring(fp.read().encode("utf-8"))
        key = filepath.parent.name
        document_text = []
        entities = []
        start = 0

        for para in xml.iter("P", "p"):
            para_text, para_entities = self._parse_paragraph(para, start=start)
            document_text.append(para_text)
            start += len(para_text)
            entities.extend(para_entities)

        document_text = "".join(document_text)
        example = {"doc_id": key, "text": document_text, "entities": entities}

        return key, example

    def _parse_example_to_kb_schema(self, example) -> Dict[str, Any]:
        text = example["text"]
        doc_id = example["doc_id"]
        passages = [
            {
                "id": f"{doc_id}-P0",
                "type": "abstract",
                "text": [text],
                "offsets": [[0, len(text)]],
            }
        ]
        entities = []
        for i, e in enumerate(example["entities"]):
            entity = {
                "id": f"{doc_id}-E{i}",
                "text": [e["phrase"]],
                "offsets": [[e["start"], e["end"]]],
                "type": e["attrs"]["type"],
                "normalized": [
                    {"db_name": "chebi", "db_id": chebi_id.strip()} for chebi_id in e["attrs"]["chebi-id"].split(",")
                ],
            }
            entities.append(entity)
        data = {
            "id": doc_id,
            "document_id": doc_id,
            "passages": passages,
            "entities": entities,
            "relations": [],
            "events": [],
            "coreferences": [],
        }
        return data

    def _read_example_from_file_in_kb_schema(self, filepath: Path) -> Tuple[str, Dict]:
        key, example = self._read_example_from_file(filepath)
        example = self._parse_example_to_kb_schema(example)
        return key, example
