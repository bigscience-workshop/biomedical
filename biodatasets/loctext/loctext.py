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
LocText is a dataset for bigbio KB schema.
This manually annotated corpus consists of 100 PubMed abstracts annotated for proteins, subcellular localizations, organisms and relations between them.
The focus of the corpus is on annotation of proteins and their subcellular localizations.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{,
  author    = {Juan Miguel Cejuela and Shrikant Vinchurkar and Tatyana Goldberg and Madhukar Sollepura Prabhu Shankar and Ashish Baghudana and Aleksandar Bojchevski and Carsten Uhlig and Andr{'{e}} Ofner and Pandu Raharja{-}Liu and Lars Juhl Jensen and Burkhard Rost},
  title     = {LocText: relation extraction of protein localizations to assist database curation},
  journal   = {BMC Bioinformatics},
  volume    = {19},
  year      = {2018},
  url       = {https://pubannotation.org/projects/LocText},
  doi       = {10.1186/s12859-018-2021-9},
  biburl    = {https://dblp.org/rec/journals/bmcbi/CejuelaVGSBBUOR18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "loctext"

_DESCRIPTION = """\
The manually annotated corpus consists of 100 PubMed abstracts annotated for proteins, subcellular localizations, organisms and relations between them.
The focus of the corpus is on annotation of proteins and their subcellular localizations.
"""

_HOMEPAGE = "https://pubannotation.org/projects/LocText"

_LICENSE = "Creative Commons Attribution 3.0 Unported License"

_URLS = {
    _DATASETNAME: "https://pubannotation.org/projects/LocText/annotations.tgz",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
    Tasks.RELATION_EXTRACTION,
]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class LocTextDataset(datasets.GeneratorBasedBuilder):
    """LocText dataset for NER, NEL and RE."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="loctext_source",
            version=SOURCE_VERSION,
            description="loctext source schema",
            schema="source",
            subset_id="loctext",
        ),
        BigBioConfig(
            name="loctext_bigbio_kb",
            version=BIGBIO_VERSION,
            description="loctext BigBio schema",
            schema="bigbio_kb",
            subset_id="loctext",
        ),
    ]

    DEFAULT_CONFIG_NAME = "loctext_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = features = datasets.Features(
                {
                    "target": datasets.Value("string"),
                    "sourcedb": datasets.Value("string"),
                    "sourceid": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "project": datasets.Value("string"),
                    "denotations": datasets.Sequence(
                        {
                            "id": datasets.Value("string"),
                            "obj": datasets.Value("string"),
                            "span": {
                                "begin": datasets.Value("int64"),
                                "end": datasets.Value("int64"),
                            },
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "id": datasets.Value("string"),
                            "obj": datasets.Value("string"),
                            "pred": datasets.Value("string"),
                            "subj": datasets.Value("string"),
                        }
                    ),
                    "namespaces": datasets.Sequence(
                        {
                            "prefix": datasets.Value("string"),
                            "uri": datasets.Value("string"),
                        }
                    ),
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
        data_dir = dl_manager.download_and_extract(urls)
        filepaths = list(Path(data_dir).glob("./LocText/*.json"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": filepaths,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepaths: List[Path], split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            for filepath in filepaths:
                key, example = self._read_example_from_file(filepath)
                yield key, example

        elif self.config.schema == "bigbio_kb":
            for filepath in filepaths:
                key, example = self._read_example_from_file_in_kb_schema(filepath)
                yield key, example

    def _read_example_from_file(self, filepath: Path) -> Tuple[str, Dict]:
        with open(filepath) as fp:
            example = json.load(fp)
        key = filepath.name.rsplit(".", 1)[0]
        # Fill default values for missing data
        for k in ["denotations", "relations"]:
            if k not in example:
                example[k] = []

        return key, example

    def _read_example_from_file_in_kb_schema(self, filepath: Path) -> Tuple[str, Dict]:
        key, example = self._read_example_from_file(filepath)
        text = example["text"]
        sourceid = example["sourceid"]
        passages = [
            {
                "id": f"{sourceid}-P0",
                "type": "abstract",
                "text": [text],
                "offsets": [[0, len(text)]],
            }
        ]
        entities = []
        for e in example["denotations"]:
            offsets = [e["span"]["begin"], e["span"]["end"]]
            entity_text = text[slice(*offsets)]
            db_name, db_id = e["obj"].split(":", 1)
            entity = {
                "id": f"{sourceid}-{e['id']}",
                "text": [entity_text],
                "offsets": [offsets],
                "type": db_name,
                "normalized": [{"db_name": db_name, "db_id": db_id}],
            }
            entities.append(entity)
        relations = []
        for r in example["relations"]:
            relation = {
                "id": f"{sourceid}-{r['id']}",
                "arg1_id": r["subj"],
                "arg2_id": r["obj"],
                "type": r["pred"],
                "normalized": [],
            }
            relations.append(relation)
        data = {
            "id": key,
            "document_id": example["sourceid"],
            "passages": passages,
            "entities": entities,
            "relations": relations,
            "events": [],
            "coreferences": [],
        }
        example = data
        return key, example


if __name__ == "__main__":
    datasets.load_dataset(__file__)
