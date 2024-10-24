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
MoNERo: a Biomedical Gold Standard Corpus for the Romanian Language for part of speech tagging and named entity recognition.
"""

import os
from pathlib import Path
from typing import Any, List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@inproceedings{,
    title = {{M}o{NER}o: a Biomedical Gold Standard Corpus for the {R}omanian Language},
    author = {Mitrofan, Maria  and Barbu Mititelu, Verginica  and Mitrofan, Grigorina},
    booktitle = "Proceedings of the 18th BioNLP Workshop and Shared Task",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5008",
    doi = "10.18653/v1/W19-5008",
    pages = "71--79",
    biburl    = {https://aclanthology.org/W19-5008.bib},
    bibsource = {https://aclanthology.org/W19-5008/}
}
"""

_DATASETNAME = "monero"

_DESCRIPTION = """\
MoNERo: a Biomedical Gold Standard Corpus for the Romanian Language for part of speech tagging and named entity recognition.
"""

_HOMEPAGE = "https://www.racai.ro/en/tools/text/"

_LICENSE = "Creative Commons License Attribution-ShareAlike 4.0 International"

_URLS = {
    # The original dataset is in 7z format hence I have downloaded and reuploded it as tar.gz format.
    # Converted via the following command:
    # curl -JLO https://www.racai.ro/media/MoNERo_2019.7z
    # mkdir -p ./MoNERo
    # pushd ./MoNERo && 7z x ../MoNERo_2019.7z && popd
    # tar -czf MoNERo.tar.gz ./MoNERo
    _DATASETNAME: "https://github.com/bigscience-workshop/biomedical/files/8550757/MoNERo.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MoneroDataset(datasets.GeneratorBasedBuilder):
    """MoNERo: a Biomedical Gold Standard Corpus for the Romanian Language for part of speech tagging and named entity recognition."""

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
                    "tokens": [datasets.Value("string")],
                    "lemmas": [datasets.Value("string")],
                    "ner_tags": [datasets.Value("string")],
                    "pos_tags": [datasets.Value("string")],
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "MoNERo", "MoNERo.txt"),
                    "split": "train",
                },
            ),
        ] + [
            datasets.SplitGenerator(
                name=split,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "MoNERo", f"MoNERo_{split}.txt"),
                    "split": split,
                },
            )
            for split in ["cardiology", "endocrinology", "diabetes"]
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            for key, example in self._read_example_from_file(filepath):
                yield key, example

        elif self.config.schema == "bigbio_kb":
            for key, example in self._read_example_from_file_in_kb_schema(filepath):
                yield key, example

    def _read_example_from_file(self, filepath: Path) -> Tuple[str, Dict]:
        with open(filepath, encoding="utf-8") as fp:
            sequences = fp.read().split("\n\n")
        for i, seq in enumerate(sequences):
            key = f"docid-{i}"
            seq = [line.rstrip().split("\t") for line in seq.rstrip().splitlines()]
            # There are few lines which only have two columns. Skipping those.
            seq = [line for line in seq if len(line) == 4]
            tokens, lemmas, ner_tags, pos_tags = zip(*seq)
            example = {
                "doc_id": key,
                "tokens": tokens,
                "lemmas": lemmas,
                "ner_tags": ner_tags,
                "pos_tags": pos_tags,
            }
            yield key, example

    @staticmethod
    def _assign_offsets(tokens):
        offsets = []
        start = 0
        for t in tokens:
            s = start
            e = s + len(t)
            offsets.append([s, e])
            start = e + 1  # Add one to include space.
        return offsets

    @staticmethod
    def _extract_entities(ner_tags):
        ner_tags = tuple(ner_tags) + ("O",)
        entities = []
        stack = []
        for i, t in enumerate(ner_tags):
            if stack and (t == "O" or t.startswith("B-")):
                l, s = stack[0]
                l, e = stack[-1]
                entity = (l, s, e)
                entities.append(entity)
                stack = []

            if t.startswith(("B-", "I-")):
                _, l = t.split("-", 1)
                stack.append((l, i))
        return entities

    def _parse_example_to_kb_schema(self, example) -> Dict[str, Any]:
        text = " ".join(example["tokens"])
        doc_id = example["doc_id"]
        offsets = self._assign_offsets(example["tokens"])
        entities_with_token_offsets = self._extract_entities(example["ner_tags"])
        passages = [
            {
                "id": f"{doc_id}-P0",
                "type": "abstract",
                "text": [text],
                "offsets": [[0, len(text)]],
            }
        ]
        entities = []
        for i, (l, s, e) in enumerate(entities_with_token_offsets):
            cs, ce = offsets[s][0], offsets[e][1]
            entity = {
                "id": f"{doc_id}-E{i}",
                "text": [text[cs:ce]],
                "offsets": [[cs, ce]],
                "type": l,
                "normalized": [],
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
        for key, example in self._read_example_from_file(filepath):
            example = self._parse_example_to_kb_schema(example)
            yield key, example
