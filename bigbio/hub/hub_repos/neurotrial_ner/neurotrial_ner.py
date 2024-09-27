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
NeuoTrialNER is an annotated dataset for named entities in clinical trial registry data in the domain of neurology/psychiatry. 
The corpus comprises 1093 clinical trial title and brief summaries from ClinicalTrials.gov. 
"""

import os
from typing import List, Tuple, Dict
import json

import datasets
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks
from .bigbiohub import kb_features

_LOCAL = False
_LANGUAGES = ['English']
_PUBMED = False
# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "neurotrial_ner"
_DISPLAYNAME = "NeuroTrialNER"
_DESCRIPTION = """\
NeuoTrialNER is an annotated dataset for named entities in clinical trial registry data in the domain of neurology/psychiatry. 
The corpus comprises 1093 clinical trial title and brief summaries from ClinicalTrials.gov. 
It has been annotated by two to three annotators for key trial characteristics, i.e., condition (e.g., Alzheimer's disease), 
therapeutic intervention (e.g., aspirin), and control arms (e.g., placebo).
"""

_HOMEPAGE = "https://github.com/Ineichen-Group/NeuroTrialNER"

_LICENSE = 'CC0_1p0'

_URL = "https://raw.githubusercontent.com/Ineichen-Group/NeuroTrialNER/main/data/annotated_data/bigbio/"
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
    "test": _URL + "test.json",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class NeuroTrialNerDataset(datasets.GeneratorBasedBuilder):
    """
    1093 clinical trial official title and brief summary from ClinicalTrials.gov
    annotated for named entities.
    """
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="neurotrial_ner_source",
            version=SOURCE_VERSION,
            description="neurotrial_ner source schema",
            schema="source",
            subset_id="neurotrial_ner",
        ),
        BigBioConfig(
            name="neurotrial_ner_bigbio_kb",
            version=BIGBIO_VERSION,
            description="neurotrial_ner BigBio schema",
            schema="bigbio_kb",
            subset_id="neurotrial_ner",
        ),
    ]

    DEFAULT_CONFIG_NAME = "neurotrial_ner_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "nctid": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokens": datasets.Value("string"),
                    "token_bio_labels": datasets.Value("string"),
                    "entities": [
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                        }
                    ],
                }
            )
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
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                    "split": "dev",
                },
            ),
        ]

    @staticmethod
    def get_source_example(uid, entry):
        nctid = entry.get("nctid", "").strip()
        text = entry.get("text", "").strip()
        tokens = entry.get("tokens", "").strip()
        token_bio_labels = entry.get("token_bio_labels", "").strip()
        entities = entry.get("entities", [])

        # Process the entities (which is a list of dictionaries)
        processed_entities = []
        for entity in entities:
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            entity_text = entity.get("text", "").strip()
            entity_type = entity.get("type", "").strip()

            processed_entities.append({
                "start": start,
                "end": end,
                "text": entity_text,
                "type": entity_type,
            })

        doc = {
            "nctid": nctid,
            "text": text,
            "tokens": tokens,
            "token_bio_labels": token_bio_labels,
            "entities": processed_entities,
        }

        return uid, doc

    @staticmethod
    def get_bigbio_example(uid, entry):
        nctid = entry.get("nctid", "").strip()
        text = entry.get("text", "").strip()
        tokens = entry.get("tokens", "").strip()
        token_bio_labels = entry.get("token_bio_labels", "").strip()
        entities = entry.get("entities", [])

        # Generate passages to capture document structure (title and brief summary)
        passages = []
        passages.append({
            "id": str(uid) + "-passage-0",
            "type": "official_title_brief_summary",
            "text": [text],
            "offsets": [[0, len(text)]],
        })

        # Process entities to conform to the schema
        processed_entities = []
        ii = 0
        for i, entity in enumerate(entities):
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            entity_text = entity.get("text", "").strip()
            entity_type = entity.get("type", "").strip()
            normalized = entity.get("normalized", [])

            processed_entities.append({
                "id": str(uid) + "-entity-" + str(ii),
                "offsets": [[start, end]],
                "text": [entity_text],
                "type": entity_type,
                "normalized": normalized,
            })
            ii += 1

        # Build the final document structure
        doc = {
            "id": uid,
            "document_id": nctid,
            "passages": passages,
            "entities": processed_entities,
            "events": [],
            "coreferences": [],
            "relations": [],
        }

        return uid, doc

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r") as f:
            data = json.load(f)
            uid = 0
            # Iterate over each entry in the JSON file
            for entry in data:
                if self.config.schema == "source":
                    yield self.get_source_example(uid, entry)
                elif self.config.schema == "bigbio_kb":
                    yield self.get_bigbio_example(uid, entry)
                uid += 1
            