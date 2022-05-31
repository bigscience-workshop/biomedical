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
Dataset containing standardised information about known adverse reactions for 200
FDA-approved drugs using information from the respective Structured Product Labels (SPLs).
This data resulted from a partnership between the United States Food and Drug Administration
(FDA) and the National Library of Medicine.

Structured Product Labels (SPLs) are the documents FDA uses to exchange information
about drugs and other products. For this dataset, SPLs were manually annotated for
adverse reactions at the mention level to facilitate development and evaluation of
text mining tools for extraction of ADRs from all SPLs. The ADRs were then normalised
to the Unified Medical Language System (UMLS) and to the Medical Dictionary for
Regulatory Activities (MedDRA).

These data were used for the adverse event challenge at TAC 2017 (Text Analysis Conference)
in four different tasks:
* Task 1: Extract AdverseReactions and related mentions (Severity, Factor, DrugClass,
Negation, Animal). This is similar to many NLP Named Entity Recognition (NER) evaluations.
* Task 2: Identify the relations between AdverseReactions and related mentions (i.e.,
Negated, Hypothetical, and Effect). This is similar to many NLP relation
identification evaluations.
* Task 3: Identify the positive AdverseReaction mention names in the labels.
For the purposes of this task, positive will be defined as the caseless strings
of all the AdverseReactions that have not been negated and are not related by
a Hypothetical relation to a DrugClass or Animal. Note that this means Factors
related via a Hypothetical relation are considered positive (e.g., "[unknown risk]
Factor of [stroke]AdverseReaction") for the purposes of this task. The result of
this task will be a list of unique strings corresponding to the positive ADRs
as they were written in the label.
* Task 4: Provide MedDRA PT(s) and LLT(s) for each positive AdverseReaction (occasionally,
two or more PTs are necessary to fully describe the reaction). For participants
approaching the tasks sequentially, this can be viewed as normalization of the terms
extracted in Task 3 to MedDRA LLTs/PTs. Because MedDRA is not publicly available,
and contains several versions, a standard version of MedDRA v18.1 will be provided
to the participants. Other resources such as the UMLS Terminology Services may be
used to aid with the normalization process.

For more information regarding the challenge at TAC 2017, please visit:
https://bionlp.nlm.nih.gov/tac2017adversereactions/

"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LOCAL = False
_CITATION = """\
@article{,
  author    = {Demner-Fushman, Dina and Shooshan, Sonya and Rodriguez, Laritza and Aronson,
               Alan and Lang, Francois and Rogers, Willie and Roberts, Kirk and Tonning, Joseph},
  title     = {A dataset of 200 structured product labels annotated for adverse drug reactions},
  journal   = {Scientific Data},
  volume    = {5},
  year      = {2018},
  month     = {01},
  pages     = {180001},
  url       = {
    https://www.researchgate.net/publication/322810855_A_dataset_of_200_structured_product_labels_annotated_for_adverse_drug_reactions
  },
  doi       = {10.1038/sdata.2018.1}
}
"""

_DATASETNAME = "spl_adr_200db"

_DESCRIPTION = """\
The United States Food and Drug Administration (FDA) partnered with the National Library
of Medicine to create a pilot dataset containing standardised information about known
adverse reactions for 200 FDA-approved drugs. The Structured Product Labels (SPLs),
the documents FDA uses to exchange information about drugs and other products, were
manually annotated for adverse reactions at the mention level to facilitate development
and evaluation of text mining tools for extraction of ADRs from all SPLs.  The ADRs were
then normalised to the Unified Medical Language System (UMLS) and to the Medical
Dictionary for Regulatory Activities (MedDRA).
"""

_HOMEPAGE = "https://bionlp.nlm.nih.gov/tac2017adversereactions/"

# NOTE: Source: https://osf.io/6h9q4/
_LICENSE = Licenses.CC0_1p0

_URLS = {
    _DATASETNAME: {
        "train": "https://bionlp.nlm.nih.gov/tac2017adversereactions/train_xml.tar.gz",
        "unannotated": "https://bionlp.nlm.nih.gov/tac2017adversereactions/unannotated_xml.tar.gz",
    }
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
    Tasks.RELATION_EXTRACTION,
]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class SplAdr200DBDataset(datasets.GeneratorBasedBuilder):
    """
    The United States Food and Drug Administration (FDA) partnered with the National Library
    of Medicine to create a pilot dataset containing standardised information about known
    adverse reactions for 200 FDA-approved drugs.

    These data were used in the adverse event challenge at TAC 2017 (Text Analysis Conference).
    For more information on the tasks, see: https://bionlp.nlm.nih.gov/tac2017adversereactions/
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    for subset_name in _URLS[_DATASETNAME]:
        BUILDER_CONFIGS.extend(
            [
                BigBioConfig(
                    name=f"spl_adr_200db_{subset_name}_source",
                    version=SOURCE_VERSION,
                    description=f"SPL ADR 200db source {subset_name} schema",
                    schema="source",
                    subset_id=f"spl_adr_200db_{subset_name}",
                ),
                BigBioConfig(
                    name=f"spl_adr_200db_{subset_name}_bigbio_kb",
                    version=BIGBIO_VERSION,
                    description=f"SPL ADR 200db BigBio {subset_name} schema",
                    schema="bigbio_kb",
                    subset_id=f"spl_adr_200db_{subset_name}",
                ),
            ]
        )

    DEFAULT_CONFIG_NAME = "spl_adr_200db_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            unannotated_features = {
                "drug_name": datasets.Value("string"),
                "text": [datasets.Value("string")],
                "sections": [
                    {
                        "id": datasets.Value("string"),
                        "name": datasets.Value("string"),
                        "text": datasets.Value("string"),
                    }
                ],
            }
            features = datasets.Features(
                {
                    **unannotated_features,
                    "mentions": [
                        {
                            "id": datasets.Value("string"),
                            "section": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "start": datasets.Value("string"),
                            "len": datasets.Value("string"),
                            "str": datasets.Value("string"),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1": datasets.Value("string"),
                            "arg2": datasets.Value("string"),
                        }
                    ],
                    "reactions": [
                        {
                            "id": datasets.Value("string"),
                            "str": datasets.Value("string"),
                            "normalizations": [
                                {
                                    "id": datasets.Value("string"),
                                    "meddra_pt": datasets.Value("string"),
                                    "meddra_pt_id": datasets.Value("string"),
                                    "meddra_llt": datasets.Value("string"),
                                    "meddra_llt_id": datasets.Value("string"),
                                    "flag": datasets.Value("string"),
                                }
                            ],
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
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        *_, subset_name = self.config.subset_id.split("_")

        urls = _URLS[_DATASETNAME][subset_name]

        data_dir = dl_manager.download_and_extract(urls)

        data_files = (Path(data_dir) / f"{subset_name}_xml").glob("*.xml")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": tuple(data_files),
                },
            ),
        ]

    def _source_features_from_xml(self, element_tree):
        root = element_tree.getroot()
        drug_name = root.attrib["drug"]

        sections = root.findall(".//Text/Section")
        relations = root.findall(".//Relations/Relation")
        reactions = [
            {
                "id": reaction.attrib["id"],
                "str": reaction.attrib["str"],
                "normalizations": [
                    {
                        # NOTE: Default features to `None` as not all of them
                        # will be present in all reactions.
                        "meddra_pt": None,
                        "meddra_pt_id": None,
                        "meddra_llt": None,
                        "meddra_llt_id": None,
                        "flag": None,
                        **normalization.attrib,
                    }
                    for normalization in reaction.findall("Normalization")
                ],
            }
            for reaction in root.findall(".//Reactions/Reaction")
        ]

        mentions = root.findall(".//Mentions/Mention")
        return {
            "drug_name": drug_name,
            "text": [section.text for section in sections],
            "mentions": [mention.attrib for mention in mentions],
            "relations": [relation.attrib for relation in relations],
            "reactions": reactions,
            "sections": [
                {**section.attrib, "text": section.text} for section in sections
            ],
        }

    def _bigbio_kb_features_from_xml(self, element_tree):
        source_features = self._source_features_from_xml(
            element_tree=element_tree,
        )
        entity_normalizations = defaultdict(list)

        for reaction in source_features["reactions"]:
            entity_name = reaction["str"]
            for normalization in reaction["normalizations"]:
                if normalization["meddra_pt_id"]:
                    entity_normalizations[entity_name].append(
                        {
                            "db_name": None,
                            "db_id": f"pt_{normalization['meddra_pt_id']}",
                        }
                    )
                if normalization["meddra_llt_id"]:
                    entity_normalizations[entity_name].append(
                        {
                            "db_name": "MedDRA v18.1",
                            "db_id": f"llt_{normalization['meddra_llt_id']}",
                        }
                    )

        section_lengths = list(
            accumulate(len(section["text"]) for section in source_features["sections"])
        )

        section_offsets = [
            (start + index, end + index)
            for index, (start, end) in enumerate(
                zip([0] + section_lengths[:-1], section_lengths)
            )
        ]

        section_start_offset_map = {
            f"S{section_index}": offsets[0]
            for section_index, offsets in enumerate(section_offsets, 1)
        }

        entities = []

        for mention in source_features["mentions"]:
            start_points = [
                int(start_point) + section_start_offset_map[mention["section"]]
                for start_point in mention["start"].split(",")
            ]

            lens = [int(len_) for len_ in mention["len"].split(",")]

            offsets = [
                (start_point, start_point + len_)
                for start_point, len_ in zip(start_points, lens)
            ]

            text = " ".join(section["text"] for section in source_features["sections"])

            entity_strings = [
                text[start_point : start_point + len_]
                for start_point, len_ in zip(start_points, lens)
            ]

            entities.append(
                {
                    "id": f"{source_features['drug_name']}_entity_{mention['id']}",
                    "type": mention["type"],
                    "text": entity_strings,
                    "offsets": offsets,
                    "normalized": entity_normalizations[mention["str"]],
                }
            )

        return {
            "document_id": source_features["drug_name"],
            "passages": [
                {
                    "id": f"{source_features['drug_name']}_section_{section['id']}",
                    "type": section["name"],
                    "text": [section["text"]],
                    "offsets": [offsets],
                }
                for section, offsets in zip(
                    source_features["sections"], section_offsets
                )
            ],
            "entities": entities,
            "relations": [
                {
                    "id": f"{source_features['drug_name']}_relation_{relation['id']}",
                    "type": relation["type"],
                    "arg1_id": relation["arg1"],
                    "arg2_id": relation["arg2"],
                    "normalized": [],
                }
                for relation in source_features["relations"]
            ],
            "events": [],
            "coreferences": [],
        }

    def _generate_examples(self, filepaths: Tuple[Path]) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        for file_index, drug_filename in enumerate(filepaths):
            element_tree = ET.parse(drug_filename)

            if self.config.schema == "source":
                features = self._source_features_from_xml(
                    element_tree=element_tree,
                )
            elif self.config.schema == "bigbio_kb":
                features = self._bigbio_kb_features_from_xml(
                    element_tree=element_tree,
                )
                features["id"] = file_index
            else:
                raise ValueError(
                    f"Unsupported schema '{self.config.schema}' requested for "
                    f"dataset with name '{_DATASETNAME}'."
                )

            yield file_index, features
