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
This dataset contains annotations for Participants, Interventions, and Outcomes (referred to as PICO task).
For 423 sentences, annotations collected by 3 medical experts are available.
To get the final annotations, we perform the majority voting.
The script loads dataset in bigbio schema (using knowledgebase schema: schemas/kb) AND/OR source (default) schema
"""
import json
from typing import Dict, List, Tuple, Union

import numpy as np

import datasets
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LOCAL = False
_CITATION = """\
@inproceedings{zlabinger-etal-2020-effective,
    title = "Effective Crowd-Annotation of Participants, Interventions, and Outcomes in the Text of Clinical Trial Reports",
    author = {Zlabinger, Markus  and
      Sabou, Marta  and
      Hofst{\"a}tter, Sebastian  and
      Hanbury, Allan},
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.274",
    doi = "10.18653/v1/2020.findings-emnlp.274",
    pages = "3064--3074",
}
"""

_DATASETNAME = "pico_extraction"

_DESCRIPTION = """\
This dataset contains annotations for Participants, Interventions, and Outcomes (referred to as PICO task).
For 423 sentences, annotations collected by 3 medical experts are available.
To get the final annotations, we perform the majority voting.
"""

_HOMEPAGE = "https://github.com/Markus-Zlabinger/pico-annotation"

_LICENSE = "Unknown"

_DATA_PATH = "https://raw.githubusercontent.com/Markus-Zlabinger/pico-annotation/master/data"
_URLS = {
    _DATASETNAME: {
        "sentence_file": f"{_DATA_PATH}/sentences.json",
        "annotation_files": {
            "intervention": f"{_DATA_PATH}/annotations/interventions_expert.json",
            "outcome": f"{_DATA_PATH}/annotations/outcomes_expert.json",
            "participant": f"{_DATA_PATH}/annotations/participants_expert.json",
        },
    }
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


def _pico_extraction_data_loader(
    sentence_file: str, annotation_files: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Dict[str, List[int]]]]]:
    """Loads four files with PICO extraction dataset:
    - one json file with sentences
    - three json files with annotations for PIO
    """
    # load sentences
    with open(sentence_file) as fp:
        sentences = json.load(fp)

    # load annotations
    annotation_dict = {}
    for annotation_type, _file in annotation_files.items():
        with open(_file) as fp:
            annotations = json.load(fp)
            annotation_dict[annotation_type] = annotations

    return sentences, annotation_dict


def _get_entities_pico(
    annotation_dict: Dict[str, Dict[str, Dict[str, List[int]]]],
    sentence: str,
    sentence_id: str,
) -> List[Dict[str, Union[int, str]]]:
    """extract entities from sentences using annotation_dict"""

    def _partition(alist, indices):
        return [alist[i:j] for i, j in zip([0] + indices, indices + [None])]

    ents = []
    for annotation_type, annotations in annotation_dict.items():
        # get indices from three annotators by majority voting
        indices = np.where(np.round(np.mean(annotations[sentence_id]["annotations"], axis=0)) == 1)[0]

        if len(indices) > 0:  # if annotations exist for this sentence
            split_indices = []
            # if there are two annotations of one type in one sentence
            for item_index, item in enumerate(indices):
                if item_index + 1 == len(indices):
                    break
                if indices[item_index] + 1 != indices[item_index + 1]:
                    split_indices.append(item_index + 1)
            multiple_indices = _partition(indices, split_indices)

            for _indices in multiple_indices:

                annotation_text = " ".join([sentence.split()[ind] for ind in _indices])

                char_start = sentence.find(annotation_text)
                char_end = char_start + len(annotation_text)

                ent = {
                    "annotation_text": annotation_text,
                    "annotation_type": annotation_type,
                    "char_start": char_start,
                    "char_end": char_end,
                }

                ents.append(ent)
    return ents


class PicoExtractionDataset(datasets.GeneratorBasedBuilder):
    """PICO Extraction dataset with annotations for
    Participants, Interventions, and Outcomes."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pico_extraction_source",
            version=SOURCE_VERSION,
            description="pico_extraction source schema",
            schema="source",
            subset_id="pico_extraction",
        ),
        BigBioConfig(
            name="pico_extraction_bigbio_kb",
            version=BIGBIO_VERSION,
            description="pico_extraction BigBio schema",
            schema="bigbio_kb",
            subset_id="pico_extraction",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pico_extraction_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
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

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "sentence_file": data_dir["sentence_file"],
                    "annotation_files": data_dir["annotation_files"],
                },
            ),
        ]

    def _generate_examples(self, split, sentence_file, annotation_files) -> (int, dict):
        """Yields examples as (key, example) tuples."""

        sentences, annotation_dict = _pico_extraction_data_loader(
            sentence_file=sentence_file, annotation_files=annotation_files
        )

        if self.config.schema == "source":
            for uid, sentence_tuple in enumerate(sentences.items()):
                sentence_id, sentence = sentence_tuple
                ents = _get_entities_pico(annotation_dict, sentence, sentence_id)

                data = {
                    "doc_id": sentence_id,
                    "text": sentence,
                    "entities": [
                        {
                            "text": ent["annotation_text"],
                            "type": ent["annotation_type"],
                            "start": ent["char_start"],
                            "end": ent["char_end"],
                        }
                        for ent in ents
                    ],
                }
                yield uid, data

        elif self.config.schema == "bigbio_kb":
            uid = 0
            for id_, sentence_tuple in enumerate(sentences.items()):
                if id_ < 2:
                    continue
                sentence_id, sentence = sentence_tuple
                ents = _get_entities_pico(annotation_dict, sentence, sentence_id)

                data = {
                    "id": str(uid),
                    "document_id": sentence_id,
                    "passages": [],
                    "entities": [],
                    "relations": [],
                    "events": [],
                    "coreferences": [],
                }
                uid += 1

                data["passages"] = [
                    {
                        "id": str(uid),
                        "type": "sentence",
                        "text": [sentence],
                        "offsets": [[0, len(sentence)]],
                    }
                ]
                uid += 1

                for ent in ents:
                    entity = {
                        "id": uid,
                        "type": ent["annotation_type"],
                        "text": [ent["annotation_text"]],
                        "offsets": [[ent["char_start"], ent["char_end"]]],
                        "normalized": [{"db_name": None, "db_id": None}],
                    }
                    data["entities"].append(entity)
                    uid += 1

                yield uid, data
