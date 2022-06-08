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
This corpus release contains 4,993 abstracts annotated with (P)articipants,
(I)nterventions, and (O)utcomes. Training labels are sourced from AMT workers and
aggregated to reduce noise. Test labels are collected from medical professionals.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import datasets
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.PICO, Tags.POS]
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{nye-etal-2018-corpus,
    title = "A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to Support Language Processing for Medical Literature",
    author = "Nye, Benjamin  and
      Li, Junyi Jessy  and
      Patel, Roma  and
      Yang, Yinfei  and
      Marshall, Iain  and
      Nenkova, Ani  and
      Wallace, Byron",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1019",
    doi = "10.18653/v1/P18-1019",
    pages = "197--207",
}
"""

_DATASETNAME = "ebm_pico"

_DESCRIPTION = """\
This corpus release contains 4,993 abstracts annotated with (P)articipants,
(I)nterventions, and (O)utcomes. Training labels are sourced from AMT workers and
aggregated to reduce noise. Test labels are collected from medical professionals.
"""

_HOMEPAGE = "https://github.com/bepnye/EBM-NLP"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://github.com/bepnye/EBM-NLP/raw/master/ebm_nlp_2_00.tar.gz"
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "2.0.0"
_BIGBIO_VERSION = "1.0.0"

PHASES = ("starting_spans", "hierarchical_labels")
LABEL_DECODERS = {
    PHASES[0]: {
        "participants": {0: "No Label", 1: "Participant"},
        "interventions": {0: "No Label", 1: "Intervention"},
        "outcomes": {0: "No Label", 1: "Outcome"},
    },
    PHASES[1]: {
        "participants": {
            0: "No label",
            1: "Age",
            2: "Sex",
            3: "Sample-size",
            4: "Condition",
        },
        "interventions": {
            0: "No label",
            1: "Surgical",
            2: "Physical",
            3: "Pharmacological",
            4: "Educational",
            5: "Psychological",
            6: "Other",
            7: "Control",
        },
        "outcomes": {
            0: "No label",
            1: "Physical",
            2: "Pain",
            3: "Mortality",
            4: "Adverse-effects",
            5: "Mental",
            6: "Other",
        },
    },
}


def _get_entities_pico(
    annotation_dict: Dict[str, List[int]],
    tokenized: List[str],
    document_content: str,
) -> List[Dict[str, Union[int, str]]]:
    """extract PIO entities from documents using annotation_dict"""

    def _partition(alist, indices):
        return [alist[i:j] for i, j in zip([0] + indices, indices + [None])]

    ents = []
    for annotation_type, annotations in annotation_dict.items():
        indices = [idx for idx, val in enumerate(annotations) if val != 0]

        if len(indices) > 0:  # if annotations exist for this sentence
            split_indices = []
            # if there are two annotations of one type in one sentence
            for item_index, item in enumerate(indices):
                if item_index + 1 == len(indices):
                    break
                if indices[item_index] + 1 != indices[item_index + 1]:
                    split_indices.append(item_index + 1)
                elif annotations[item] != annotations[item + 1]:
                    split_indices.append(item_index + 1)
            multiple_indices = _partition(indices, split_indices)

            for _indices in multiple_indices:
                high_level_type = LABEL_DECODERS["starting_spans"][annotation_type][1]
                fine_grained_type = LABEL_DECODERS["hierarchical_labels"][
                    annotation_type
                ][annotations[_indices[0]]]
                annotation_text = " ".join([tokenized[ind] for ind in _indices])

                char_start = document_content.find(annotation_text)
                char_end = char_start + len(annotation_text)

                ent = {
                    "annotation_text": annotation_text,
                    "high_level_annotation_type": high_level_type,
                    "fine_grained_annotation_type": fine_grained_type,
                    "char_start": char_start,
                    "char_end": char_end,
                }

                ents.append(ent)
    return ents


class EbmPico(datasets.GeneratorBasedBuilder):
    """A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to
    Support Language Processing for Medical Literature."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="ebm_pico_source",
            version=SOURCE_VERSION,
            description="ebm_pico source schema",
            schema="source",
            subset_id="ebm_pico",
        ),
        BigBioConfig(
            name="ebm_pico_bigbio_kb",
            version=BIGBIO_VERSION,
            description="ebm_pico BigBio schema",
            schema="bigbio_kb",
            subset_id="ebm_pico",
        ),
    ]

    DEFAULT_CONFIG_NAME = "ebm_pico_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "text": datasets.Value("string"),
                            "annotation_type": datasets.Value("string"),
                            "fine_grained_annotation_type": datasets.Value("string"),
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
        else:
            raise ValueError("config.schema must be either source or bigbio_kb")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        documents_folder = Path(data_dir) / "ebm_nlp_2_00" / "documents"
        annotations_folder = (
            Path(data_dir) / "ebm_nlp_2_00" / "annotations" / "aggregated"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "documents_folder": documents_folder,
                    "annotations_folder": annotations_folder,
                    "split_folder": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "documents_folder": documents_folder,
                    "annotations_folder": annotations_folder,
                    "split_folder": "test/gold",
                },
            ),
        ]

    def _generate_examples(
        self, documents_folder, annotations_folder, split_folder: str
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        annotation_types = ["interventions", "outcomes", "participants"]

        docs_path = os.path.join(
            annotations_folder,
            f"hierarchical_labels/{annotation_types[0]}/{split_folder}/",
        )
        documents_in_split = os.listdir(docs_path)

        uid = 0
        for id_, document in enumerate(documents_in_split):
            document_id = document.split(".")[0]
            with open(f"{documents_folder}/{document_id}.tokens") as fp:
                tokenized = fp.read().splitlines()
            document_content = " ".join(tokenized)

            annotation_dict = {}
            for annotation_type in annotation_types:
                try:
                    with open(
                        f"{annotations_folder}/hierarchical_labels/{annotation_type}/{split_folder}/{document}"
                    ) as fp:
                        annotation_dict[annotation_type] = [
                            int(x) for x in fp.read().splitlines()
                        ]
                except OSError:
                    annotation_dict[annotation_type] = []

            ents = _get_entities_pico(
                annotation_dict, tokenized=tokenized, document_content=document_content
            )

            if self.config.schema == "source":

                data = {
                    "doc_id": document_id,
                    "text": document_content,
                    "entities": [
                        {
                            "text": ent["annotation_text"],
                            "annotation_type": ent["high_level_annotation_type"],
                            "fine_grained_annotation_type": ent[
                                "fine_grained_annotation_type"
                            ],
                            "start": ent["char_start"],
                            "end": ent["char_end"],
                        }
                        for ent in ents
                    ],
                }
                yield id_, data

            elif self.config.schema == "bigbio_kb":
                data = {
                    "id": str(uid),
                    "document_id": document_id,
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
                        "type": "document",
                        "text": [document_content],
                        "offsets": [[0, len(document_content)]],
                    }
                ]
                uid += 1

                for ent in ents:
                    entity = {
                        "id": uid,
                        "type": f'{ent["high_level_annotation_type"]}_{ent["fine_grained_annotation_type"]}',
                        "text": [ent["annotation_text"]],
                        "offsets": [[ent["char_start"], ent["char_end"]]],
                        "normalized": [],
                    }
                    data["entities"].append(entity)
                    uid += 1

                yield uid, data
